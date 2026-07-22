"""Spells cast into the world as short-lived, self-animating mobs.

A spell is an :class:`~.entity.Entity` that lives on one maze, draws its own
sprites straight into the scene's 'actors' layer, carries one or more point
lights, and advances in real time on its own daemon thread -- like a torch's
flame (see :class:`~.item.Torch`), it is game state that ticks on a background
clock and pushes each change out through ``scene.request_redraw()``. A spell is
*not* part of the turn system: it does not wait for the player to move, and it
never enters the maze inventory, so it neither blocks movement nor is "walked
on" -- it is pure spectacle that lights the room while it lasts.

Because a spell never joins the inventory, its light cannot follow a host
location the usual way; instead each light is *pinned* to a maze cell (see
:meth:`.light.Light.pin`). A moving spell re-pins its light every step. Pinning
drops the light's own caches but does not tell the level its lighting is stale,
so after moving (or after building/tearing down the lights) the spell nudges
the level to recomposite and repaint -- the same signal the flicker thread uses.

Two spells ship today, both cast in a chosen cardinal direction:

- :class:`Fireball` -- a single very bright, slightly-white point light (about
  ten times a torch) that flies ``*`` across the map until it strikes a wall.
- :class:`Lightning` -- a string of ``/-\\|`` symbols that snaps into being
  along an almost-straight path, each symbol carrying a fierce cold-white point
  light, and vanishes again in under a second.

Game-side module: no rendering library may be imported here.
"""
import threading
import time

import numpy as np

from .entity import Entity
from .heat import Heat
from .light import PointLight
from .location import Location
from .units import lm


#: Arrow-key name -> unit ``(dx, dy)`` on the maze, matching the movement
#: handler's convention: x grows rightward, y grows upward (the maze image is
#: loaded flipped, so "up" is +y). The cast prompt collects one arrow key and
#: the interpreter maps it through here.
DIRECTIONS = {
    'Right': (1, 0),
    'Left': (-1, 0),
    'Up': (0, 1),
    'Down': (0, -1),
}


class Spell(Entity):
    """Base for spells: entity/location plumbing, pinned lights, the anim thread.

    Subclasses build their sprites and lights in ``__init__`` and implement
    :meth:`_run` (the body of the animation thread) and :meth:`_teardown_sprites`.
    The base owns being an entity with a (deliberately empty) location the
    lights can hang off, the list of pinned lights, and the one-shot teardown
    that removes every light and asks the level to repaint.
    """

    #: kelvin dumped into a wall this spell strikes; None for a spell that
    #: leaves no heat behind (see :meth:`_strike`).
    STRIKE_TEMP = None

    def __init__(self, scene, maze, entity_type, obj_name=None):
        Entity.__init__(self, entity_type=entity_type, obj_name=obj_name)
        self.scene = scene
        self.maze = maze
        # whether this spell runs its animation thread; the heat it leaves on a
        # strike inherits it, so a headless test spell spawns no heat thread.
        self._animated = False
        # A location for the lights to subscribe to, left empty forever: the
        # spell pins its lights to cells directly and never registers in the
        # maze inventory, so it stays invisible to walkability / on_walked_on.
        self.location = Location(self, None, None)
        self.lights = []
        self.sprite = None
        self._done = False
        self._thread = None

    # -- lights ------------------------------------------------------------

    def _add_light(self, pos, color, brightness=1.0):
        """Create a point light pinned to maze cell *pos* ``(x, y)``."""
        light = PointLight(self, color=color, brightness=brightness)
        light.pin(self.maze, tuple(int(v) for v in pos))
        self.lights.append(light)
        return light

    def _relight(self):
        """Recomposite this spell's level and ask for a repaint.

        Pinning (or adding) a light does not on its own mark the level's
        lighting stale; firing one light's ``changed`` signal drives the level
        the same way the torch flicker thread does.
        """
        if self.lights:
            self.lights[0].changed()

    def _in_bounds(self, pos):
        """True if maze cell *pos* ``(x, y)`` lies within the maze."""
        x, y = int(pos[0]), int(pos[1])
        h, w = self.maze.shape
        return 0 <= x < w and 0 <= y < h

    def _open(self, pos):
        """True if maze cell *pos* ``(x, y)`` is in bounds and walkable.

        A spell stops (or is snuffed) at anything a walker could not stand on,
        which is how a fireball finds a wall and a bolt stops arcing into rock.
        """
        if not self._in_bounds(pos):
            return False
        x, y = int(pos[0]), int(pos[1])
        return bool(self.maze.blocktype_at(y, x)['walkable'])

    # -- heat --------------------------------------------------------------

    def _strike(self, cell):
        """Dump this spell's :data:`STRIKE_TEMP` heat into the struck wall *cell*.

        Creates a :class:`~.heat.Heat` on the cell that glows and cools on its
        own; it outlives the spell. The new mob animates only when this spell
        does (``start``), so a headless, tick-driven spell in a test leaves the
        heat inert for the test to drive rather than spawning a thread. Ignores
        an out-of-bounds *cell* (a bolt reaching the map edge).
        """
        if self.STRIKE_TEMP is None or not self._in_bounds(cell):
            return None
        return Heat(self.scene, self.maze, cell, temperature=self.STRIKE_TEMP,
                    start=self._animated)

    # -- lifecycle ---------------------------------------------------------

    def _start(self):
        self._thread = threading.Thread(target=self._run, name=str(self.type),
                                        daemon=True)
        self._thread.start()

    def _run(self):
        raise NotImplementedError()

    def _teardown_sprites(self):
        if self.sprite is not None:
            self.scene.sprite_layers['actors'].remove_sprites(self.sprite)
            self.sprite = None

    def destroy(self):
        """Remove the spell from the world: lights out, sprites gone, repaint.

        Idempotent -- the animation thread and an external caller may both
        reach it. Runs on whatever thread calls it (the spell's own thread when
        the effect ends naturally); it only mutates game-side state and fires
        thread-safe signals.
        """
        if self._done:
            return
        self._done = True
        for light in self.lights:
            light.destroy()
        level = self.maze.level
        if level is not None:
            level.invalidate_lighting()
            level.lighting_changed()
        self._teardown_sprites()


class Fireball(Spell):
    """A blazing point light that flies ``*`` in a straight line until a wall.

    One very bright, slightly-white point light (about ten times a torch, its
    blue lifted so the core reads warm-white rather than orange) rides a single
    ``*`` sprite. It advances at :data:`SPEED` cells/second, one cell at a time
    so a wall is never stepped over, and snuffs itself the instant the next cell
    is not walkable.
    """

    GLYPH = '*'
    #: Luminous flux per channel, in lumens (see :class:`~.light.PointLight`).
    #: ~10x a torch (``Torch.LIGHT_COLOR`` = (15, 12, 3)) with the blue channel
    #: lifted so the fierce core reads warm-white, not orange.
    COLOR = (150*lm, 135*lm, 75*lm)
    FG = (1.0, 0.95, 0.8, 1.0)
    SPEED = 30.0            # cells / second
    STEP_INTERVAL = 1 / 60.
    #: the wall it hits glows fresh-ember hot (a full-brightness orange glow)
    STRIKE_TEMP = 1000.0

    def __init__(self, scene, maze, pos, direction, start=True):
        Spell.__init__(self, scene, maze, entity_type='mob.spell.fireball',
                       obj_name='fireball')
        self._animated = start
        self.heat = None
        self.direction = np.array(direction, dtype=int)
        self.pos = np.array(pos, dtype=int)
        self._float_pos = self.pos.astype(float)

        self.sprite = scene.sprite_layers['actors'].add_sprites((1,))
        self.sprite.glyph = scene.glyphs[self.GLYPH]
        self.sprite.fgcolor = self.FG
        self.sprite.bgcolor = (0, 0, 0, 0)

        self._add_light(self.pos, self.COLOR)
        self._draw()
        self._relight()
        if start:
            self._start()

    def _draw(self):
        x, y = int(self.pos[0]), int(self.pos[1])
        # zval below the player (-0.2) but above items/mobs (-0.1) so the bolt
        # reads clearly as it passes over them
        self.sprite.position = (float(x), float(y), -0.15)

    def _advance_to(self, pos):
        self.pos = pos
        self.lights[0].pin(self.maze, tuple(int(v) for v in pos))
        self._draw()
        self._relight()

    def advance(self, dt):
        """Fly the bolt forward *dt* seconds; snuff it on a wall.

        Walks one cell at a time toward the accumulated float position so a
        fast bolt on a long frame cannot leap over a wall. Returns False (and
        destroys the bolt) the moment the next cell is not walkable, True while
        it still lives. Injectable ``dt`` so tests drive travel without sleeping.
        """
        self._float_pos += self.direction * self.SPEED * dt
        while not self._done:
            step = np.clip(np.trunc(self._float_pos - self.pos), -1, 1).astype(int)
            if not step.any():
                break
            nxt = self.pos + step
            if not self._open(nxt):
                self.heat = self._strike(nxt)   # leave the wall glowing
                self.destroy()
                return False
            self._advance_to(nxt)
        return not self._done

    def _run(self):
        last = time.perf_counter()
        while not self._done:
            time.sleep(self.STEP_INTERVAL)
            now = time.perf_counter()
            dt, last = now - last, now
            self.advance(dt)


class Lightning(Spell):
    """A bolt of ``/-\\|`` symbols that strobes in 2-3 quick flashes, then dies.

    The bolt is traced :data:`LENGTH` cells from the caster in the chosen
    direction, veering one cell sideways now and then (:data:`WANDER`) so the
    path is almost -- but not quite -- straight. Each cell gets its own ``/-\\|``
    symbol and a fierce cold-white point light. Rather than shine steadily, the
    whole bolt strobes: it snaps on and off :data:`FLASH_RANGE` times, each
    flash lit for :data:`ON_TIME` and dark for :data:`OFF_TIME`, then is torn
    down. Flashing reuses the same lights and sprites -- brightness and sprite
    visibility toggle in place -- so no light churns on and off the level.
    """

    LENGTH = 10
    #: Luminous flux per channel, in lumens (see :class:`~.light.PointLight`).
    #: Fierce cold white: blue >= green > red, and brighter than a fireball.
    COLOR = (180*lm, 240*lm, 360*lm)
    FG = (0.85, 0.92, 1.0, 1.0)
    #: how many quick flashes, picked per cast from ``randint(*FLASH_RANGE)``
    #: (2 or 3); each flash is ON_TIME lit + OFF_TIME dark (whole strobe < 1s).
    FLASH_RANGE = (2, 4)
    ON_TIME = 0.2          # seconds a flash stays lit
    OFF_TIME = 0.1         # seconds of dark between flashes
    WANDER = 0.35           # chance a segment veers one cell sideways
    #: a bolt is far hotter than a fireball -- it superheats the wall it earths
    #: into to blue-white, then cools down through the whole blackbody ramp
    STRIKE_TEMP = 3000.0

    def __init__(self, scene, maze, pos, direction, start=True):
        Spell.__init__(self, scene, maze, entity_type='mob.spell.lightning',
                       obj_name='lightning')
        self._animated = start
        # the wall cell the bolt earths into, if it is stopped by one; set by
        # _trace and turned into a glowing Heat once the lights are built
        self._impact = None
        cells, glyphs = self._trace(np.array(pos, dtype=int),
                                    np.array(direction, dtype=int))
        self.cells = cells

        self.sprite = scene.sprite_layers['actors'].add_sprites((len(cells),))
        self.sprite.glyph = np.array([scene.glyphs[g] for g in glyphs],
                                     dtype='uint32')
        self.sprite.fgcolor = self.FG
        self.sprite.bgcolor = (0, 0, 0, 0)
        # the lit positions, restored on each flash-on; a flash-off hides the
        # sprites (NaN) so the glyphs blink in step with their lights
        self._positions = np.array([(c[0], c[1], -0.15) for c in cells],
                                   dtype='float32')
        self.sprite.position = self._positions

        for c in cells:
            self._add_light(c, self.COLOR)
        self._relight()
        # leave the earthed-into wall glowing (outlives the brief bolt)
        self.heat = self._strike(self._impact) if self._impact is not None else None
        if start:
            self._start()

    def _set_lit(self, on):
        """Show or hide the whole bolt in place (brightness + sprite position).

        Toggling brightness drives each light's ``changed`` signal, so the level
        recomposites and repaints exactly as it does for a flickering flame; the
        sprites are moved off-screen (NaN) while dark so the glyphs blink too.
        """
        if self.sprite is None:
            return
        for light in self.lights:
            light.brightness = 1.0 if on else 0.0
        self.sprite.position = (self._positions if on
                                else np.full_like(self._positions, np.nan))

    def _run(self):
        for _ in range(int(np.random.randint(*self.FLASH_RANGE))):
            self._set_lit(True)
            time.sleep(self.ON_TIME)
            if self._done:
                return
            self._set_lit(False)
            time.sleep(self.OFF_TIME)
            if self._done:
                return
        self.destroy()

    def _trace(self, origin, direction):
        """Return ``(cells, glyphs)`` for an almost-straight bolt from *origin*.

        Steps one cell along *direction* per segment, occasionally adding a
        sideways nudge; each segment's glyph follows the direction it was drawn
        in. Stops early if the path runs into a wall.
        """
        perp = np.array([-direction[1], direction[0]])
        cells = [tuple(int(v) for v in origin)]
        glyphs = [self._glyph(direction)]
        cur = origin.copy()
        for _ in range(self.LENGTH - 1):
            step = direction.copy()
            if np.random.rand() < self.WANDER:
                step = step + perp * np.random.choice([-1, 1])
            nxt = cur + step
            if not self._open(nxt):
                self._impact = tuple(int(v) for v in nxt)
                break
            cells.append(tuple(int(v) for v in nxt))
            glyphs.append(self._glyph(step))
            cur = nxt
        return cells, glyphs

    @staticmethod
    def _glyph(step):
        """The ``/-\\|`` symbol for a one-cell *step*.

        Cardinal steps are ``-``/``|``; a diagonal is ``/`` when it rises to the
        right (dx*dy > 0, with y up) and ``\\`` when it falls to the right.
        """
        dx, dy = int(step[0]), int(step[1])
        if dx and dy:
            return '/' if dx * dy > 0 else '\\'
        return '-' if dx else '|'


#: Spell name -> factory ``(scene, maze, pos, direction) -> Spell``. The cast
#: prompt matches typed text against these names; "bal" -> fireball, "lit" ->
#: lightning (see dialogs.cast).
SPELLS = {
    'fireball': Fireball,
    'lightning': Lightning,
}
