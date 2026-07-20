# coding: utf8
import numpy as np

from .layers import GlyphRegistry, SpriteLayer, FieldLayer, LayerList
from .maze import Maze
from .array_cache import ArraySumCache
from .entity import Entity
from .events import Observable
from .world import SIGHT_SUPERSAMPLE, Level


class MessageLog(object):
    """Messages shown to the user, held as game state (``scene.log``).

    Follows the layer change-tracking contract: ``version`` bumps on every
    mutation and the ``changed`` Observable is invoked after each bump. Game
    threads write messages; a game-side painter (hud.py) renders the tail into
    a CharGridLayer — nothing rendering-side reads the log directly.
    """
    def __init__(self):
        self.lines = []
        self.version = 0
        self.changed = Observable()

    def write(self, text):
        """Append *text* to the log, splitting on newlines."""
        self.lines.extend(text.split('\n'))
        self._changed()

    def set_last_line(self, line):
        """Replace the last line (command-prompt editing)."""
        self.lines[-1] = line
        self._changed()

    def remove_last_line(self):
        self.lines.pop(-1)
        self._changed()

    def _changed(self):
        self.version += 1
        self.changed()


class Screen(object):
    """The canvas size in character cells (rows, cols), held as game state.

    The display backend writes it on window resize; game-side painters
    (hud.Hud) observe it and re-lay out their grids. Follows the layer
    change-tracking contract (``version`` + ``changed`` Observable). The
    default matches the default 1400x900 window at (10, 16)-pixel cells, so
    headless code sees the standard layout.
    """
    def __init__(self, shape=(56, 140)):
        self.shape = tuple(shape)
        self.version = 0
        self.changed = Observable()

    def set_shape(self, shape):
        """Record a new cell shape; no-op (and no bump) when unchanged."""
        shape = tuple(shape)
        if shape == self.shape:
            return
        self.shape = shape
        self.version += 1
        self.changed()


class Scene(Entity):
    """Game state: the landscape, player, items, and mobs.

    Contains no rendering code. What should be displayed is written into
    game-owned render layers (see layers.py):

    - ``glyphs`` and ``sprite_layers`` ('scenery', 'items', 'actors') carry
      character sprites; entities write into them as they move.
    - ``sight`` is a FieldLayer holding the composited lighting * line-of-sight
      + memory field, updated by update_sight().

    A rendering backend (e.g. backends.vispy.VispySceneRenderer) consumes
    these layers and injects ``visibility``: an object with
    ``render(pos, read=True) -> (h, w, >=3) array`` that computes a shadow map
    for a light/viewer at a maze position.
    """

    # sight memory fades to this fraction of itself per second (equivalent to
    # the historical 0.999-per-frame decay at 60 fps)
    MEMORY_DECAY_RATE = 0.999 ** 60

    def __init__(self):
        Entity.__init__(self, entity_type='scene')
        self._player = None

        # game-owned render layers: the bridge between game state and renderers
        self.glyphs = GlyphRegistry()
        self.sprite_layers = {name: SpriteLayer(name) for name in ('scenery', 'items', 'actors')}

        # screen-space CharGridLayers (menus, pagers, console/HUD text); a
        # backend renders each entry generically, in list order
        self.grids = LayerList()

        # messages to be shown to the user; hud.ConsolePainter renders these
        self.log = MessageLog()

        # canvas size in cells, backend-written; the HUD lays out against it
        self.screen = Screen()

        # subscribed to by the display backend; see request_redraw()
        self.redraw_requested = Observable()

        # invoked (no arguments) after set_level() has swapped in a new maze
        # and resized the sight fields. Subscribers rebuild whatever they
        # sized from the old maze -- the vispy backend rebuilds its
        # ShadowRenderer and sight texture. Fired on the calling (game)
        # thread, after all of the scene's own state is consistent.
        self.level_changed = Observable()

        # visibility provider (see class docstring); injected by the rendering
        # backend or by headless test code
        self.visibility = None

        # set by the gameplay input handler (Escape); the display backend
        # polls it from its frame tick and shuts down
        self.quit_requested = False

        self.supersample = SIGHT_SUPERSAMPLE

        self.norm_light = None
        self.light_cache = ArraySumCache()

        # Divisor that maps the composited log-lighting into 0-1. Held across
        # frames so that a flickering light actually modulates the output:
        # renormalising per frame would divide the flicker straight back out
        # (the peak would pin to 1.0 no matter how the flame behaved). Cleared
        # whenever the lights themselves move.
        self._light_norm = None

        # composited sight field consumed by renderers. Kept as one object for
        # the scene's lifetime -- set_level() reshapes it in place (FieldLayer
        # .set_data reallocates on a shape change) so a backend's reference
        # stays valid across level switches.
        self.sight = FieldLayer('sight', shape=(0, 0, 3))

        # the visible Level; set by set_level(). The maze and the sight fields
        # are read through it (see the properties below) rather than copied
        # onto the scene, so "the current level's field" is never a separate
        # fact that can fall out of step with the current level.
        self._level = None
        self._scenery = None

        # track all items
        self.items = []

        # track monsters by location
        self.monsters = {}

        self._need_los_update = True

        # the multi-level world, installed by set_world(). Until then the scene
        # runs on a single unnamed maze, which is what the tests and the
        # screenshot harness want -- they place things at dungeon coordinates
        # and never travel.
        self.world = None

        self.set_level(Maze.load_image('level1.png'))

    def set_world(self, world):
        """Install *world* and switch to its current level.

        The scene delegates nothing else to the world: it still holds exactly
        one visible maze. What the world adds is the ability to ask *which
        level is this* and *what portal is underfoot* (see dm.DungeonMaster).
        """
        self.world = world
        self.set_level(world.current)

    @property
    def level(self):
        """The Level currently displayed."""
        return self._level

    # The level owns these; the scene reads them through whichever level is
    # current. They are properties rather than copies so that there is exactly
    # one place each value lives -- copying them onto the scene is what let a
    # maze and a sight field of the wrong size be observed together.
    @property
    def maze(self):
        return self._level.maze

    @property
    def field_shape(self):
        return self._level.field_shape

    @property
    def memory(self):
        return self._level.memory

    @property
    def line_of_sight(self):
        return self._level.line_of_sight

    def set_level(self, level):
        """Make *level* visible, rebuilding everything sized from its maze.

        *level* may be a :class:`~.world.Level` or a bare Maze, which is
        wrapped in an anonymous Level -- so the scene always has a level, and
        the delegating properties above always have somewhere to point. It is
        the single entry point for level initialization: called once from
        __init__ for the starting level, and again for every later switch.

        Callers must move the player onto the new level themselves; the scene
        does not place entities.
        """
        if not isinstance(level, Level):
            level = level.level or Level(level.obj_name or 'level', level,
                                         supersample=self.supersample)

        # the outgoing level has no viewer any more, so nothing on it is in
        # sight; its memory is left alone, being what the player remembers
        if self._level is not None and self._level is not level:
            self._level.clear_line_of_sight()

        self._level = level
        if self.world is not None and level.name in self.world.levels:
            self.world.current = level

        # rebuild the scenery sprites: free the outgoing maze's slot before
        # allocating the new one, so the layer does not grow by a whole maze
        # on every switch.
        scenery_layer = self.sprite_layers['scenery']
        if self._scenery is not None:
            scenery_layer.remove_sprites(self._scenery)
        self._scenery = level.maze.add_scenery(self.glyphs, scenery_layer)

        # FieldLayer.set_data reallocates on a shape change, so scene.sight
        # keeps its identity for backends holding it.
        self.sight.set_data(np.zeros(level.field_shape, dtype='float32'))

        # every lighting cache is sized by, or positioned against, the maze
        self.norm_light = None
        self._light_norm = None
        self.light_cache = ArraySumCache()
        self._need_los_update = True

        self.level_changed()

    @property
    def player(self):
        return self._player

    @player.setter
    def player(self, player):
        assert self._player is None
        self._player = player
        player.location.global_changed.connect(self._player_moved)
        self._player_moved()

    def _player_moved(self, event=None):
        self._need_los_update = True
        self.norm_light = None  # should have a more intelligent way to clear this cache
        self._light_norm = None

    def monster_moved(self, mon, old_pos):
        if old_pos is not None:
            self.monsters[tuple(old_pos)].remove(mon)
        self.monsters.setdefault(tuple(mon.position), []).append(mon)

    def add_item(self, item):
        self.items.append(item)

    def write(self, message):
        """Display a message to the user."""
        self.log.write(message)

    def items_at(self, pos):
        """Return the items lying in the maze at *pos*."""
        return [e for e in self.maze.inventory[tuple(pos)] if e.type.isa('item')]

    def request_redraw(self):
        """Ask the display to repaint.

        Entities call this when they change something the display derives but
        does not observe directly -- lighting, most notably, which is
        recomputed during the draw itself and so cannot be an observed layer.
        Follows the observer contract used by the render layers: the backend
        subscribes to ``redraw_requested`` with a callback that only sets a
        dirty flag, so this is safe to call from any thread.
        """
        self.redraw_requested()

    def invalidate_lighting(self):
        """Discard the composited lighting; it is rebuilt on the next draw.

        Called by light sources whose emitted light changed (brightness,
        colour). Moving a light additionally invalidates the per-light shadow
        maps, which the item handles itself.
        """
        self.norm_light = None

    def update_sight(self, dt):
        """Advance the sight/memory field by *dt* seconds and write the result
        into the ``sight`` FieldLayer.

        Called once per rendered frame by the display backend. LOS and lighting
        are recomputed only when invalidated (player moved, lights changed).

        Everything here is read off one Level, captured once: the fields being
        composited and the lights contributing to them then necessarily belong
        to the same level and are all the same shape.
        """
        level = self._level

        # render new line of sight
        if self._need_los_update:
            level.line_of_sight = self.player.line_of_sight().astype('float32', copy=False)
            self._need_los_update = False
        line_of_sight = level.line_of_sight

        # calculate lighting. Only this level's lights are consulted, so their
        # maps are all sized to this level -- no light on another level can
        # contribute a differently-shaped array to the sum.
        if self.norm_light is None:
            lights = []
            for light in level.lights:
                light_map = light.lightmap(supersample=self.supersample)
                if light_map is None:
                    continue
                lights.append(light_map)
            if lights:
                # ArraySumCache.sum_arrays asserts on an empty list, and a
                # level may legitimately hold no light at all
                light = self.light_cache.sum_arrays(lights)
                log_light = np.log(np.clip(light*10, 1, np.inf))
                if self._light_norm is None:
                    self._light_norm = log_light.max()
                self.norm_light = log_light / self._light_norm
            else:
                self.norm_light = np.zeros(level.field_shape, dtype='float32')

        # current sight is combination of lighting and LOS
        sight = line_of_sight * self.norm_light

        self.sight.set_data(level.memory * (1 - line_of_sight) + sight)

        # forget
        level.memory *= self.MEMORY_DECAY_RATE ** dt

        # add sight to memory. Reducing the length-3 trailing axis with
        # sight.max(axis=2) is ~20x slower than maximum() applied pairwise.
        brightest = np.maximum(np.maximum(sight[:, :, 0], sight[:, :, 1]), sight[:, :, 2])
        level.memory[:, :, 2] = np.maximum(level.memory[:, :, 2], brightest)
