"""The :class:`Spell` base: entity plumbing, pinned lights, the anim thread.

Game-side module: no rendering library may be imported here.
"""
import threading

from ..entity import Entity
from ..heat import Heat
from ..light import PointLight
from ..location import Location


class Spell(Entity):
    """Base for spells: entity/location plumbing, pinned lights, the anim thread.

    Subclasses build their sprites and lights in ``__init__`` and implement
    :meth:`_run` (the body of the animation thread) and :meth:`_teardown_sprites`.
    The base owns being an entity with a (deliberately empty) location the
    lights can hang off, the list of pinned lights, and the one-shot teardown
    that removes every light and asks the level to repaint.
    """

    #: Whether the cast prompt collects a direction before this spell is built.
    #: Every world-cast projectile does; a self-cast spell that acts on the
    #: player alone (see :data:`.SPELLS`, ``glo``) overrides this to False so the
    #: prompt fires the instant its name resolves.
    NEEDS_DIRECTION = True

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
