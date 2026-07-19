# Architecture: game state ⇄ renderer bridge

The game model is renderer-agnostic. Everything a display needs is published
through a small set of game-owned, numpy-only data structures ("layers"), and
everything the game needs back from the display is a single injected object
(the visibility provider) plus a per-frame clock tick. Dialogs, the HUD, and
the message console are all game state too — screen-space glyph grids — so
a second rendering backend (terminal, a different GPU API, ...) implements
only the consumer side of one pipeline, never game logic.

```
 game objects           game state              glyph layers              backends/vispy/
 ─────────────────      ──────────────────      ─────────────────────     ─────────────────────
 Player, Item,       →  Scene: maze,         →  GlyphLayer (base)      →  window.py   (canvas,
 Monster, DM,            sprite_layers,          ├─ SpriteLayer           camera, frame tick)
 CommandInterpreter,     grids (LayerList),       │    (sparse, world)   render.py   (sprite
 dialogs/ (Menu,         log (MessageLog),       └─ CharGridLayer         layers -> GL)
 Pager, DialogSession)   glyphs, sight            (dense, screen)      grids.py    (generic
                                                 FieldLayer 'sight'         CharGridLayer -> GL)
                     ←  scene.visibility                              ←  graphics.py (visuals,
                     ←  scene.update_sight(dt)                            CharAtlas, GL shadow
                                                                           renderer)
                                                                        input.py    (canvas keys
                                                                           -> InputEvent)
```

The render side never knows what it is drawing: menus, the console, and the
maze are all just versioned collections of glyphs that get lit, filtered, and
sent to shaders. All meaning — what a menu is, what a cursor row looks like,
what the HUD says — lives on the game side, written into the glyph layers by
game-side painters.

## Module boundary

**One rule**: only modules under `carriage_return/backends/` may import
vispy, Qt, or OpenGL. Every other module in `carriage_return/` — including
the `dialogs/` subpackage and `interpreter.py` — is game state and must stay
renderer-agnostic.

`tests/test_boundaries.py` enforces this by *discovering* the game-side
module list (walking `carriage_return/` and skipping `backends/`, rather than
maintaining a hand-written list — new modules are covered automatically),
importing each one in a subprocess, and asserting no forbidden module ended
up in `sys.modules`. `tests/test_scene.py::test_game_model_is_headless`
additionally runs the game model end-to-end (via
`agent_helpers/check_headless.py`) with a numpy-only visibility provider and
the real dialog pipeline (reading a scroll opens a pager) with no rendering
library imported.

## The layer bridge (`carriage_return/layers.py`)

Change tracking everywhere is by integer version counters, not events: a
write costs one numpy slice assignment plus an increment; backends compare
counters once per frame and re-upload only what changed. A static scene costs
a backend a few integer compares and zero copies/uploads per frame.

Every layer kind shares one base, `GlyphLayer`, which owns the contract:

```
GlyphLayer            name, version, structure_version, changed, _changed()
├── CharGridLayer     dense rows×cols glyph/fg/bg arrays, space + anchor
│                       screen space: menus, pagers, console/HUD text
└── SpriteLayer       sparse per-glyph positions via slot handles, world space
                        scenery, items, actors — anything positioned freely
```

- ``version`` bumps on any data write; ``structure_version`` additionally
  bumps when the underlying arrays are reallocated (new sprites/slots added,
  a grid's membership in `scene.grids` changing), meaning references a
  backend holds into the old arrays/list are stale.
- ``changed`` is an `events.Observable` invoked (with no arguments) after
  any version bump. Any number of components may `connect()` to it. An
  interactive backend subscribes a cheap, thread-safe "schedule a frame"
  function (the vispy backend connects its dirty-flag setter,
  `MainWindow.mark_dirty`) so game-state changes repaint without polling;
  sync still happens at draw time by diffing versions. Do **not**
  observe the `sight` FieldLayer from a draw-scheduling callback — it is
  recomputed during every draw, which would schedule draws forever.

### GlyphRegistry (`scene.glyphs`)

Append-only mapping char → small int glyph id, in insertion order.
`registry[char]` adds missing chars. Backends map ids to their own
representation (atlas index, terminal character) by consuming
`registry.chars` incrementally when `registry.version` changes. A backend
that feeds `registry.chars` to a `CharAtlas` in registry order gets an
**identity id → atlas index mapping** (this mirrors `CharAtlas.add_chars`,
including its unconditional re-add of duplicate chars — do not deduplicate
on either side without translating ids). Both consumers in this repo
(`backends/vispy/render.py`'s `VispyLayerRenderer.sync` and
`backends/vispy/grids.py`'s `GridRenderer.sync`) document the same
consumed-count rule: advance the "already synced" counter by exactly the
number of characters just consumed (`len(new_chars)`), never by
`len(registry.chars)` — game threads (dialog threads, the gamepad thread)
can append to the registry concurrently between the slice and the update, and
using the live length would silently skip those characters from the atlas
forever. This was a real bug during the split, not a hypothetical one.

### SpriteLayer / SpriteSlot (`scene.sprite_layers`: `scenery`, `items`, `actors`)

The sparse `GlyphLayer`: every sprite has its own free (x, y, z) position.
Owns contiguous arrays shared by all its slots: `position` float32 (N,3),
`glyph` uint32 (N,), `fgcolor`/`bgcolor` float32 (N,4). `add_sprites(shape)`
returns a `SpriteSlot` handle; entities write through slot property setters
(scalars/arrays, numpy broadcasting; `None` writes NaN). Conventions:

- **NaN position = hidden sprite** (`SingleCharSprite.hide()` uses this).
- Draw order/occlusion between sprites comes from the z coordinate of
  positions, not layer identity.
- Writes must go through the setters; mutating `slot.position[...]` in place
  bypasses version tracking.

Per-glyph positioning stays for *all* world content on purpose — scenery does
not migrate to a dense grid, even though it is static, because free
per-glyph positions are reserved for future visual effects. Dense
`CharGridLayer`s are for screen-space text content only.

### CharGridLayer (`scene.grids`, dialog/HUD grids)

The dense `GlyphLayer`: a rows×cols block of character cells.
`space='screen'` grids anchor to the canvas (menus, pagers, the console and
HUD boxes) — the only space in use; `space='world'` is reserved for grids in
maze coordinates and nothing constructs one. `anchor` names a canvas edge or
corner (`'center'`, `'top'`, ..., `'bottom-right'`) plus an optional `offset`
cell displacement. A grid always renders at one cell per `char_size` pixels
— never squeezed to fit. Dialog grids keep their construction shape (a grid
larger than the canvas overflows the edge); the HUD grids are reshaped by
their painters when the window size changes (below).

```python
grid.write(row, col, text, fg=None, bg=None)   # clipped; one version bump
grid.fill_row(row, fg=None, bg=None)           # cursor highlight bars
grid.clear(fg=None, bg=None)                   # reset to spaces
grid.reshape(shape)                            # reallocate blank; structure bump
```

Window resizes reach the game through `scene.screen` (a `Screen`: the canvas
size in cells, written by the backend from the resize event). The `Hud`
observes it, reshapes its three grids to the new width — stats spanning the
full width, info/console splitting the bottom rows — and repaints them with
text re-wrapped. `reshape` bumps `structure_version`, which tells the
backend's grid visual to refit (sprite region, camera rect, placement) at
its next sync.

All meaning — borders, titles, hint lines, checkbox glyphs, cursor bars — is
written into the grid as characters and cell colors by game-side painters
(`dialogs/base.py`'s `CharGridPainter`, `hud.py`'s painters). The renderer
(`backends/vispy/grids.py`) draws a rectangle of cells and nothing else; it
has no idea a border or a cursor exists.

`scene.grids` is a `LayerList`: an ordered, versioned collection.
List order is draw order among screen-space grids (later = on top);
`structure_version` bumps on add/remove so a backend can diff membership at
sync time without polling every grid.

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
`ShadowRenderer` (`backends/vispy/graphics.py`, geometry-shader shadow
volumes rendered to an FBO and read back) is the production implementation;
`tests/test_scene.py::FakeVisibility` is the trivial numpy one. A CPU
shadowcaster is the intended future alternative for headless/terminal use —
`Maze.opacity` (and `Maze.opaque_geometry()`) are the canonical inputs.
Consumers: `Player.line_of_sight()` and `Item.shadow_map()` (light sources).

## Input (`carriage_return/input.py`, game-side)

`input.py` carries no rendering/GUI import; the vispy keyboard shim lives at
`backends/vispy/input.py`, the only place that ever sees a native vispy key
event.

### Event classes

One base class (shared repr for debugging) with one subclass per kind;
consumers match by `isinstance`, never on type strings:

```python
InputEvent            # base
├── KeyPress(key, text=None)     # key: plain string name; text: typed chars
├── KeyRelease(key, text=None)   # same fields
├── GamepadEvent(state)          # state: dict of evdev-style codes -> values
├── FocusIn                      # delivered on handler-stack changes, below
├── FocusOut
└── Close                        # DialogSession.get() raises DialogClosed
```

Key normalization (native key object → plain string name) happens once, at
the backend boundary, in `backends/vispy/input.py`; game-side code only ever
sees strings.

### Dispatcher and handlers

```python
class InputDispatcher:
    """Stack of handlers; events offered top-down until one returns True."""
    def add_handler(self, h):     # notifies displaced top: h_old.handle(FocusOut())
    def remove_handler(self, h):  # notifies newly exposed top: handle(FocusIn())
    def dispatch(self, event):
        for handler in reversed(list(self.handlers)):
            if handler.handle(event) is True:
                return handler
```

- No internal lock: list append/remove and the `list(...)` snapshot are
  atomic under the GIL, and a handler removed mid-dispatch receiving one
  last event is fine — handlers are queues.
- A handler is modal simply by returning `True` for everything;
  `QueuedInputHandler` does exactly that, so a `DialogSession` pushed onto
  the stack *is* capturing input, with no separate modal mechanism.
- The dispatcher always iterates the whole stack top-down; nothing asks who
  is "the top handler".
- There is no key-grab table. On any stack change the dispatcher delivers
  `FocusOut` to the handler that lost the top and `FocusIn` to the one that
  gained it; stateful handlers (`GameplayInputHandler`) clear held-key state
  on `FocusOut`. Trade-off: a movement key held across a dialog must be
  re-pressed afterwards — standard roguelike behavior.

The contract: **`handle(event)` only routes** — it must be cheap,
thread-safe, and must never mutate game state. It runs on whatever thread the
input source lives on (the GUI thread for keyboard events, the gamepad
thread for pads). Handlers that do real work are queue-backed:

```python
class QueuedInputHandler(InputHandler):
    """Consumes everything into a queue.Queue drained by the owner's thread."""
    def handle(self, event):
        self.queue.put(event)
        return True
```

`GameplayInputHandler` and `CommandInputHandler` are both `QueuedInputHandler`
subclasses with their own daemon thread, exactly like a `DialogSession`:
**input-source threads never mutate game state.**

- `GameplayInputHandler`: no vispy timer, no dispatch decisions. Its loop
  blocks on the queue (or wakes on a movement-repeat timeout while a
  direction key is held) and handles key presses — `t`/`r`/`d` call
  `interpreter.take([])`/`read([])`/`drop([])`, `Tab` toggles the command
  prompt, `Escape` sets `scene.quit_requested`. While a dialog sits above it
  on the stack it receives nothing and holds no keys (cleared by
  `FocusOut`), so it simply blocks — suspension is a *consequence of the
  stack*, not a decision the handler makes. The clock is injectable
  (`clock=time.monotonic`) and `_step`/`_process` are separate methods so
  movement-repeat timing is testable without sleeping.
- `CommandInputHandler`: also its own thread; prompt editing targets
  `scene.log` (`set_last_line`/`write`/`remove_last_line`) — no vispy
  anywhere in the path.
- `DialogSession` (`dialogs/session.py`): the same pattern — see Dialogs,
  below.
- `GamepadThread`: a plain `threading.Thread` (only needs the `inputs`
  package), calling `dispatcher.dispatch(GamepadEvent(state))` directly —
  handlers enqueue, so no marshalling back to a GUI thread is needed.
- `backends/vispy/input.py`'s `CanvasInputSource`: the vispy keyboard shim,
  the only input code with a vispy import — connects canvas key events,
  normalizes them, and calls `dispatcher.dispatch(...)`.

## Threading model

Two rules:

1. **Game state is mutated only by handler-owned threads** — the gameplay
   thread, dialog threads, the command-prompt thread, plus their callees (DM,
   monsters, painters). The modal stack (§ Input) guarantees exactly one of
   them is receiving input — and therefore mutating game state — at a time,
   without any lock.
2. **The GUI thread only reads**, version-gated, at frame time. Callbacks
   connected by the backend do nothing but set a dirty flag (an atomic write,
   safe from any thread, no Qt involved).

Redraw scheduling: the backend installs `mark_dirty` (`MainWindow.mark_dirty`,
just `self._dirty = True`) to the `changed` event of every layer/grid/log
it watches. `MainWindow`'s 60 Hz timer is the **frame tick**
(`MainWindow._frame_tick`): scroll the camera, sync the grid renderer, and —
if `_dirty` — clear the flag and call `canvas.update()`; then check
`scene.quit_requested` and close the canvas if it was set. A burst of writes
from any number of threads collapses into at most one repaint per tick.

Known, accepted looseness (unchanged since before this redesign): the
renderer may read a half-finished turn (player moved, monster not yet). If
that ever shows visibly, the fix is a scene-level "turn" version/snapshot —
out of scope here.

The never-observe-`sight` rule from the layer bridge section is part of the
same discipline: `sight` is recomputed on every draw
(`VispySceneRenderer.update`), so connecting `mark_dirty` to it would
schedule a draw from inside every draw, forever.

## Dialogs (`carriage_return/dialogs/`, game-side)

A dialog is entirely game state: its display is a screen-space
`CharGridLayer` in `scene.grids` (written by a painter) and its input is the
dispatcher stack — the `DialogSession` *is* the top handler while it's open.
Opening a dialog therefore needs no UI and works headless by construction —
`open_menu(scene, ...)` runs the real pipeline with nothing attached to a
display (see `tests/test_actions.py`, `agent_helpers/check_headless.py`).

```
carriage_return/dialogs/
    __init__.py       # public API: open_menu(), open_pager(), DialogSession, DialogClosed
    session.py         # DialogSession (queue + thread + finished event)
    base.py             # Widget base (version/changed/done/result), CharGridPainter base
    menu.py              # Menu, MenuItem, run_menu(), MenuPainter
    pager.py             # Pager, run_pager(), PagerPainter    ("book")
```

### Lifecycle

```python
def open_menu(scene, title, items, multi_select=False) -> DialogSession:
    menu = Menu(title, _as_menu_items(items), multi_select)
    painter = MenuPainter(scene, menu)     # builds + paints the grid, synchronously
    return _run_dialog(lambda s: run_menu(s, menu), painter, title)

def _run_dialog(body, painter, name):
    session = DialogSession(body, name=name)
    session.activate()                     # push onto the InputDispatcher stack
    def teardown(_):                       # dialog thread; game state only
        session.deactivate()
        painter.close()
    session.finished.connect(teardown)
    return session.start()                 # spawns the dialog thread, returns immediately
```

`open_pager(scene, title, pages)` mirrors this with `Pager`/`run_pager`/
`PagerPainter`. The painter is constructed (and paints once) on the calling
thread *before* the dialog thread starts, so the grid is present in
`scene.grids` as soon as `open_menu`/`open_pager` returns — useful for
deterministic screenshots and tests that don't want to wait on a thread.

Callers get results via `session.finished.connect(cb)` — `cb(session)` runs
on the dialog thread; `session.result` is `None` on cancel.
`session.finished` is an `events.OneShotEvent` (see below); there is no
GUI-thread result variant, because nothing about finishing a dialog needs the
GUI thread — the renderer notices the grid disappeared at its next frame
tick.

### `OneShotEvent` (`carriage_return/events.py`)

Replaces hand-rolled `_callbacks` + `_lock` + `threading.Event` bookkeeping
with one reusable class: fires exactly once with a value, guarantees every
callback runs. `connect(cb)` runs `cb(value)` immediately, on the caller's
thread, if the event already fired; otherwise it registers `cb` to run when
`fire(value)` is called (on the firing thread). An internal lock makes the
registered-vs-fired race impossible without exposing a mutex to callers.
`DialogSession.finished` is one; exceptions from callbacks propagate rather
than being swallowed.

### Painters (`dialogs/base.py::CharGridPainter`)

Content-layout logic (border, padding, title, cursor bar, hint line) lives in
one base class; subclasses declare a content shape and fill it:

```python
class MenuPainter(CharGridPainter):          # dialogs/menu.py
    def _content_shape(self): ...            # rows/cols the content needs
    def _render_content(self): ...           # title, item rows, cursor bar, hints
```

A painter connects itself to its model's `changed` event and repaints the whole
grid on any change — pure numpy, so it is correct to run on the dialog
thread (the model's sole mutator while it's active); the grid's version bump
is what wakes the renderer. Painters own their own colors (border/title/hint
colors distinct from the HUD's — see `CharGridPainter.FG`/`BG` etc.); the
renderer never special-cases them.

### `DialogSession` (`dialogs/session.py`)

A `QueuedInputHandler` subclass: a dialog body runs as a plain sequential
function on its own daemon thread, pulling `InputEvent`s with `session.get()`
(which raises `DialogClosed` on a `Close` event, so every dialog loop unwinds
correctly without each author checking for it) and mutating its widget model
in a loop. `session.finished` (a `OneShotEvent`) fires with the session
itself when the body returns; any other exception is recorded as
`session.error` and re-raised in the dialog thread (surfaces via
`threading.excepthook` rather than being silently swallowed).

### Legacy console letter-menu

There is exactly one selection UI. Typed `take`/`drop` commands with
ambiguous targets open the same modal `Menu` the `t`/`d` shortcuts do (see
Action layer, below) — there is no separate letter-selection flow.

## Action layer (`carriage_return/interpreter.py`, game-side)

`CommandInterpreter` is the single action layer for both input paths — the
`t`/`r`/`d` gameplay shortcuts and typed console commands call the same
methods:

```python
class CommandInterpreter:
    def __init__(self, scene): ...        # player, log, dialogs all via scene

    def take(self, args):
        items = self.scene.items_at(player.location.slot)
        # none -> message; unambiguous -> self._take(items); ambiguous ->
        session = dialogs.open_menu(self.scene, "Take which items?", items, multi_select=True)
        session.finished.connect(lambda s: s.result and self._take(s.result))

    def read(self, args): ...   # readables triage -> player.read(item)
    def drop(self, args): ...   # menu over inventory -> player.drop(item)
```

`Scene` keeps only world-state queries (`items_at`), the layers/grids/log,
`update_sight`, and `write` — it does not carry out actions or open dialogs.
Reading belongs to the player: `Player.read(item)` calls `item.read(self)`,
so in-game constraints and consequences (a scroll opening its own pager) live
on the item:

```python
class Scroll(Item):
    pages = [...]
    def read(self, reader):
        return dialogs.open_pager(self.scene, self.description, self.pages)
```

## HUD (`carriage_return/hud.py`, game-side)

Every fixed text panel — console, stats bar, info box — is a screen-space
`CharGridLayer` filled by a game-side painter; a rendering backend draws them
exactly like any other grid, with no idea any of them is "the HUD":

- `ConsolePainter(scene, shape)` — observes `scene.log` (a `MessageLog`:
  `lines`, `version`, `changed`, `write`/`set_last_line`/`remove_last_line`)
  and repaints its visible tail into a bottom-right-anchored grid, newest
  line last, character-wrapping lines wider than the box. The command prompt
  reaches it through `scene.log.set_last_line`, so `CommandInputHandler`
  needs no vispy either.
- `StaticTextPainter(scene, shape, text, anchor, ...)` — a fixed block of
  text in a grid, wrapped to the box width; used for the stats bar (a
  full-width, one-text-row band above the boxes) and the info box
  (bottom-left).
- `build_hud(scene)` constructs all three as a `Hud` (`.info`, `.console`,
  `.stats`, `.close()`). The `Hud` derives every shape from `scene.screen`
  and subscribes to its `changed` event: a resize reshapes all three grids (stats
  spanning the full width, info/console splitting the bottom rows 40/60)
  and repaints them with text re-wrapped.

All three painters draw a one-cell `+--+` border ring (`dialogs.base
.draw_border`, shared with the dialog painters) and fill their grid's
`bgcolor` with a translucent dark cell background (`(0, 0, 0, 0.4)`) via
`grid.clear(fg=..., bg=...)`, so HUD text stays legible over a lit or
textured maze background instead of drawing on the grid's default
transparent-black cells. Dialog painters set their own, different colors
(`dialogs/base.py::CharGridPainter`) and are untouched by this.

`scene.messages` (a vispy-style `EventEmitter`) does not exist; `scene.log`
is the one mechanism for game-to-display messages.

## The vispy backend (`carriage_return/backends/vispy/`)

All vispy/Qt/OpenGL code lives under this one package (see Module boundary,
above):

```
carriage_return/backends/vispy/
    __init__.py   # re-exports MainWindow, VispySceneRenderer, GridRenderer,
                  #   CanvasInputSource for convenient wiring
    window.py     # MainWindow: canvas, cameras, frame tick (scroll + grid
                  #   sync + dirty-check + quit check)
    render.py     # VispySceneRenderer / VispyLayerRenderer: sprite-layer
                  #   sync, GL shadow renderer wiring, sight texture upload
    grids.py      # GridRenderer: generic CharGridLayer renderer — one visual
                  #   per screen-space scene.grids entry, created/destroyed
                  #   by diffing LayerList.structure_version, uploaded
                  #   version-gated; carries no model knowledge (a border is
                  #   just cells to it)
    graphics.py   # visuals (SpritesVisual/Sprites), CharAtlas, GL shadow
                  #   renderer (ShadowRenderer), shaders/filters
    input.py      # CanvasInputSource: canvas key events -> normalized
                  #   InputEvent -> dispatcher.dispatch(); the only input
                  #   code with a vispy import
```

`MainWindow` knows nothing about dialog/HUD/console content — `TextBox` and
`Console` widgets do not exist; every text panel renders through
`grids.py`'s generic renderer. `MainWindow.attach_scene(scene)` only starts
the `GridRenderer` over `scene.grids` and returns; dialogs, the command
interpreter, and input handlers are all game-side and constructed by the
caller (`return_to_carriage.py`).

Wiring, end to end:

```python
scene = Scene(); dm = DungeonMaster(scene); player = Player(scene)
hud = build_hud(scene)
interp = CommandInterpreter(scene)
dispatcher = InputDispatcher()

ui = MainWindow(dispatcher)                     # backend: canvas + input source
renderer = VispySceneRenderer(ui, scene)        # backend: sprite/shadow rendering
ui.attach_scene(scene)                          # backend: generic grid renderer

cmd_input_handler = CommandInputHandler(scene.log, interp)
gameplay = GameplayInputHandler(dm, player, interpreter=interp,
                                command_handler=cmd_input_handler)
gameplay.activate()
ui.follow_entity(player)
```

## Frame protocol

- The backend calls `scene.update_sight(dt)` once per rendered frame
  (dt in seconds), from `VispySceneRenderer._on_draw`. LOS recomputes only
  when the player moved; lighting only when invalidated; memory decays by
  `MEMORY_DECAY_RATE ** dt` (time-based, equivalent to the historical
  0.999/frame at 60 fps).
- Sprite-layer sync (`VispyLayerRenderer.sync`) is version-gated and runs
  inside the sprites visual's `_prepare_draw`, so it also covers offscreen
  `SceneCanvas.render()` calls, which do not emit `canvas.events.draw`.
  Grid sync (`GridRenderer.sync`) is likewise version-gated but is *not*
  hooked to a visual — `MainWindow._frame_tick` calls it explicitly every
  tick, and batch/screenshot code must call `ui.grid_renderer.sync()`
  itself (see `agent_helpers/render_screenshot.py`).
- Game state is discrete (per-turn); smooth animation is the renderer's
  business (e.g. camera scrolling lives in `backends/vispy/window.py`).
  Don't put per-frame cosmetic state into the layers.
- Known limitation (inherited from the pre-split design): the vispy backend
  runs `update_sight` as a canvas *draw-event* callback, i.e. after the scene
  drew, so a sight change appears one frame late.

## What a second backend implements

1. Consume `scene.glyphs` + `scene.sprite_layers` + `scene.grids` (all
   version-gated; `scene.grids` additionally diffed by
   `structure_version` for add/remove).
2. Consume `scene.sight` (version-gated) and apply it as a mask.
3. Provide `scene.visibility` (GL, CPU shadowcaster, or trivial).
4. Call `scene.update_sight(dt)` once per frame.
5. Render `scene.log`'s tail *only if* it writes its own HUD — normally it
   doesn't need to, since `hud.py`'s `ConsolePainter` already turns the log
   into a `scene.grids` entry that step 1 covers.
6. Feed normalized `InputEvent`s (see `input.py`) to an `InputDispatcher`,
   with key/gamepad normalization happening once at the backend boundary
   (see `backends/vispy/input.py` for the reference implementation). No
   dialog-specific code is needed — dialogs are just more `scene.grids`
   entries plus handlers already on the dispatcher stack.

## Verification

- `.../envs/rtc/bin/python -m pytest tests/ -q` — includes the boundary and
  headless gates.
- `agent_helpers/check_take_read.py <menu.png> <pager.png>` — end-to-end
  check of the take/read gameplay dialogs through the real input path
  (canvas key events -> `CanvasInputSource` -> `InputDispatcher` ->
  `GameplayInputHandler` thread -> `CommandInterpreter` ->
  `dialogs.open_menu`/`open_pager` -> `DialogSession` on the stack -> dialog
  thread), asserting on inventory, the map, and `scene.log`, and saving a
  screenshot at each dialog's open state.
- Screenshot regression (deterministic: fixed `np.random.seed`, fixed `dt`),
  two baselines in `agent_helpers/` (gitignored, regenerated locally — no
  PNGs are committed):

  ```sh
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
      python agent_helpers/render_screenshot.py out.png          # base frame
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
      python agent_helpers/render_screenshot.py out_menu.png --menu  # take menu open

  python agent_helpers/compare_screenshots.py agent_helpers/baseline.png out.png
  python agent_helpers/compare_screenshots.py agent_helpers/baseline_menu.png out_menu.png
  ```

  The `--menu` frame opens the take menu the same way `check_take_read.py`
  does, but by calling `CommandInterpreter.take([])` directly rather than
  driving canvas key events — `open_menu`'s painter paints synchronously on
  the calling thread, so the grid is present with no GUI event pump or
  timing dependency, keeping the baseline deterministic. It exercises the
  generic grid renderer (`backends/vispy/grids.py`) end-to-end.

  The `__NV_PRIME_*` variables (the `nvidia` shell alias) are required on
  this machine — Qt aborts on GLX otherwise.

- `agent_helpers/render_resized.py <out.png> <width> <height>` — same scene
  after a window resize; verifies the resize path (HUD grids reshaped and
  re-wrapped via `scene.screen`, no compressed text, no overlap). Not a
  stored baseline; inspect the output.
