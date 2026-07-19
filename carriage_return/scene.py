# coding: utf8
import numpy as np

from .layers import GlyphRegistry, SpriteLayer, FieldLayer, LayerList
from .maze import Maze
from .array_cache import ArraySumCache
from .entity import Entity
from .events import Observable


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

        # create maze
        self.maze = Maze.load_image('level1.png')

        # add scenery sprites for drawing maze
        self.maze.add_scenery(self.glyphs, self.sprite_layers['scenery'])

        # visibility provider (see class docstring); injected by the rendering
        # backend or by headless test code
        self.visibility = None

        # set by the gameplay input handler (Escape); the display backend
        # polls it from its frame tick and shuts down
        self.quit_requested = False

        ms = self.maze.shape
        self.supersample = 4
        self.field_shape = (ms[0] * self.supersample, ms[1] * self.supersample, 3)

        self.norm_light = None
        self.light_cache = ArraySumCache()

        # Divisor that maps the composited log-lighting into 0-1. Held across
        # frames so that a flickering light actually modulates the output:
        # renormalising per frame would divide the flicker straight back out
        # (the peak would pin to 1.0 no matter how the flame behaved). Cleared
        # whenever the lights themselves move.
        self._light_norm = None

        self.memory = np.zeros(self.field_shape, dtype='float32')
        self.line_of_sight = np.zeros(self.field_shape, dtype='float32')

        # composited sight field consumed by renderers
        self.sight = FieldLayer('sight', shape=self.field_shape)

        # track all items
        self.items = []

        # track monsters by location
        self.monsters = {}

        self._need_los_update = True

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
        """
        # render new line of sight
        if self._need_los_update:
            self.line_of_sight = self.player.line_of_sight().astype('float32', copy=False)
            self._need_los_update = False

        # calculate lighting
        if self.norm_light is None:
            lights = []
            for item in self.items:
                if not item.light_source:
                    continue
                item_visible = True  # todo
                if not item_visible:
                    continue
                item_light = item.lightmap(supersample=self.supersample)
                if item_light is None:
                    continue
                lights.append(item_light)
            light = self.light_cache.sum_arrays(lights)
            log_light = np.log(np.clip(light*10, 1, np.inf))
            if self._light_norm is None:
                self._light_norm = log_light.max()
            self.norm_light = log_light / self._light_norm

        # current sight is combination of lighting and LOS
        sight = self.line_of_sight * self.norm_light

        self.sight.set_data(self.memory * (1 - self.line_of_sight) + sight)

        # forget
        self.memory *= self.MEMORY_DECAY_RATE ** dt

        # add sight to memory. Reducing the length-3 trailing axis with
        # sight.max(axis=2) is ~20x slower than maximum() applied pairwise.
        brightest = np.maximum(np.maximum(sight[:, :, 0], sight[:, :, 1]), sight[:, :, 2])
        self.memory[:, :, 2] = np.maximum(self.memory[:, :, 2], brightest)
