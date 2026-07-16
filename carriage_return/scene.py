# coding: utf8
import numpy as np

from .events import EventEmitter
from .layers import GlyphRegistry, SpriteLayer, FieldLayer
from .maze import Maze
from .array_cache import ArraySumCache
from .entity import Entity


class Scene(Entity):
    """Game state: the landscape, player, items, and mobs.

    Contains no rendering code. What should be displayed is written into
    game-owned render layers (see layers.py):

    - ``glyphs`` and ``sprite_layers`` ('scenery', 'items', 'actors') carry
      character sprites; entities write into them as they move.
    - ``sight`` is a FieldLayer holding the composited lighting * line-of-sight
      + memory field, updated by update_sight().

    A rendering backend (e.g. render_vispy.VispySceneRenderer) consumes these
    layers and injects ``visibility``: an object with
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

        # create maze
        self.maze = Maze.load_image('level1.png')

        # add scenery sprites for drawing maze
        self.maze.add_scenery(self.glyphs, self.sprite_layers['scenery'])

        # visibility provider (see class docstring); injected by the rendering
        # backend or by headless test code
        self.visibility = None

        ms = self.maze.shape
        self.supersample = 4
        self.field_shape = (ms[0] * self.supersample, ms[1] * self.supersample, 3)

        self.norm_light = None
        self.light_cache = ArraySumCache()

        self.memory = np.zeros(self.field_shape, dtype='float32')
        self.line_of_sight = np.zeros(self.field_shape, dtype='float32')

        # composited sight field consumed by renderers
        self.sight = FieldLayer('sight', shape=self.field_shape)

        # messages to be shown to the user; UI code connects to this
        self.messages = EventEmitter(source=self, type='message')

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

    def monster_moved(self, mon, old_pos):
        if old_pos is not None:
            self.monsters[tuple(old_pos)].remove(mon)
        self.monsters.setdefault(tuple(mon.position), []).append(mon)

    def add_item(self, item):
        self.items.append(item)

    def write(self, message):
        """Display a message to the user."""
        self.messages(message=message)

    def request_player_action(self, action):
        if action == 'take':
            items = self.items_at(self.player.location.slot)
            if len(items) == 0:
                self.write("Nothing to take here.")
            else:
                for item in items:
                    self.player.take(item)
                    self.write("Taken: %s" % item.name)
        elif action == 'read':
            self.player.read_item()

    def user_request_item(self, message, items, callback):
        """Ask the user to select an item from a list.
        """
        self.write(message)
        raise NotImplementedError("interactive item selection not implemented")

    def update_sight(self, dt):
        """Advance the sight/memory field by *dt* seconds and write the result
        into the ``sight`` FieldLayer.

        Called once per rendered frame by the display backend. LOS and lighting
        are recomputed only when invalidated (player moved, lights changed).
        """
        # render new line of sight
        if self._need_los_update:
            self.line_of_sight = self.player.line_of_sight()
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
            self.norm_light = log_light / log_light.max()

        # current sight is combination of lighting and LOS
        sight = self.line_of_sight * self.norm_light

        self.sight.set_data(self.memory * (1 - self.line_of_sight) + sight)

        # forget
        self.memory *= self.MEMORY_DECAY_RATE ** dt

        # add sight to memory
        self.memory[:, :, 2] = np.maximum(self.memory[:, :, 2], sight.max(axis=2))
