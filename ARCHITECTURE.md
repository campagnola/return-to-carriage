# Architecture: game state ⇄ renderer bridge

The game model is renderer-agnostic. Everything a display needs is published
through a small set of game-owned, numpy-only data structures ("layers"), and
everything the game needs back from the display is a single injected object
(the visibility provider) plus a per-frame clock tick. A second rendering
backend (terminal, different GPU API, ...) implements the consumer side of
this contract without touching game code.

```
 game state (vispy-free)        bridge                 vispy/OpenGL side
 ─────────────────────────      ──────────────────     ─────────────────────────
 Scene, Maze, Player, Item,  →  GlyphRegistry          render_vispy.py
 Monster, Location,             SpriteLayers            (VispySceneRenderer)
 Inventory, DungeonMaster       FieldLayer 'sight'     graphics.py (visuals,
                             ←  scene.visibility        ShadowRenderer)
                             ←  scene.update_sight(dt) ui.py, input.py,
                             →  scene.messages          interpreter.py
```

## Module boundary

Game-side modules — everything in `carriage_return/` except `ui.py`,
`input.py`, `interpreter.py`, `graphics.py`, `render_vispy.py` — must not
import vispy, Qt, or OpenGL. `tests/test_boundaries.py` enforces this by
importing every game-side module in a subprocess and checking `sys.modules`;
`tests/test_scene.py::test_game_model_is_headless` additionally runs the game
model end-to-end (via `agent_helpers/check_headless.py`) with a numpy-only
visibility provider.

## The layer bridge (`carriage_return/layers.py`)

Change tracking everywhere is by integer version counters, not events: a
write costs one numpy slice assignment plus an increment; backends compare
counters once per frame and re-upload only what changed. A static scene costs
a backend a few integer compares and zero copies/uploads per frame.

### GlyphRegistry (`scene.glyphs`)

Append-only mapping char → small int glyph id, in insertion order.
`registry[char]` adds missing chars. Backends map ids to their own
representation (atlas index, terminal character) by consuming
`registry.chars` incrementally when `registry.version` changes. A backend
that feeds `registry.chars` to a `CharAtlas` in registry order gets an
**identity id → atlas index mapping** (this mirrors `CharAtlas.add_chars`,
including its unconditional re-add of duplicate chars — do not deduplicate
on either side without translating ids).

### SpriteLayer / SpriteSlot (`scene.sprite_layers`: `scenery`, `items`, `actors`)

A SpriteLayer owns contiguous arrays shared by all its slots:
`position` float32 (N,3), `glyph` uint32 (N,), `fgcolor`/`bgcolor` float32
(N,4). `add_sprites(shape)` returns a SpriteSlot handle; entities write
through slot property setters (scalars/arrays, numpy broadcasting; `None`
writes NaN). Conventions:

- **NaN position = hidden sprite** (`SingleCharSprite.hide()` uses this).
- Draw order/occlusion between layers comes from the z coordinate of
  positions, not layer identity.
- `layer.version` bumps on any setter write. `layer.structure_version`
  additionally bumps when the arrays are reallocated (slot added/reshaped) —
  any references a backend holds into the old arrays are then stale.
- Writes must go through the setters; mutating `slot.position[...]` in place
  bypasses version tracking.

### FieldLayer (`scene.sight`)

A named float32 array plus `version` (`set_data()` copies in place and bumps;
`bump()` declares an in-place mutation). Field shape is
`(maze_h * supersample, maze_w * supersample, 3)` with `scene.supersample = 4`;
`scene.field_shape` is authoritative. `sight` holds the fully composited
visibility field: `memory * (1 - line_of_sight) + lighting * line_of_sight`,
normalized log-scaled lighting summed over light-source items (with
`ArraySumCache` reuse). Backends apply it as a per-cell brightness/color
mask over the sprites (the vispy backend uploads it to a texture and attaches
`TextureMaskFilter`; a terminal backend could threshold it into
visible/remembered/dark).

## Visibility provider (`scene.visibility`, injected)

The one service the game needs from outside:

```python
provider.render(pos, read=True) -> ndarray (h, w, >=3), values 0..255
```

— a shadow/visibility map for a viewer or light source at maze position
`pos`, at `field_shape` resolution (white = unoccluded). The GL
`ShadowRenderer` (graphics.py, geometry-shader shadow volumes rendered to an
FBO and read back) is the production implementation;
`tests/test_scene.py::FakeVisibility` is the trivial numpy one. A CPU
shadowcaster is the intended future alternative for headless/terminal use —
`Maze.opacity` (and `Maze.opaque_geometry()`) are the canonical inputs.
Consumers: `Player.line_of_sight()` and `Item.shadow_map()` (light sources).

## Frame protocol

- The backend calls `scene.update_sight(dt)` once per rendered frame
  (dt in seconds). LOS recomputes only when the player moved; lighting only
  when invalidated; memory decays by `MEMORY_DECAY_RATE ** dt`
  (time-based, equivalent to the historical 0.999/frame at 60 fps).
- Sprite-layer sync is version-gated. The vispy backend runs it inside the
  visual's `_prepare_draw` so it also covers offscreen
  `SceneCanvas.render()` calls, which do not emit `canvas.events.draw`.
- Game state is discrete (per-turn); smooth animation is the renderer's
  business (e.g. camera scrolling lives in `ui.py`). Don't put per-frame
  cosmetic state into the layers.
- Known limitation (inherited from the pre-split design): the vispy backend
  runs `update_sight` as a canvas *draw-event* callback, i.e. after the scene
  drew, so a sight change appears one frame late.

## Messages

`Scene.write(msg)` emits on `scene.messages` (`carriage_return/events.py`
EventEmitter; handler receives an event with `.message`). UI code subscribes
(`scene.messages.connect(lambda ev: ui.console.write(ev.message))` in
`return_to_carriage.py`). Note: unlike vispy's emitter, callback exceptions
propagate.

## What a second backend implements

1. Consume `scene.glyphs` + `scene.sprite_layers` (version-gated).
2. Consume `scene.sight` (version-gated) and apply it as a mask.
3. Provide `scene.visibility` (GL, CPU shadowcaster, or trivial).
4. Call `scene.update_sight(dt)` once per frame.
5. Subscribe to `scene.messages`; deliver player input by calling game
   methods (see `input.py` for the vispy/Qt reference implementation).

## Verification

- `.../envs/rtc/bin/python -m pytest tests/ -q` — includes the boundary and
  headless gates.
- Screenshot regression (deterministic, compares against a captured
  baseline):

  ```sh
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
      python agent_helpers/render_screenshot.py out.png
  python agent_helpers/compare_screenshots.py baseline.png out.png
  ```

  The `__NV_PRIME_*` variables (the `nvidia` shell alias) are required on
  this machine — Qt aborts on GLX otherwise.
