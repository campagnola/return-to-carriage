# coding: utf8
import numpy as np
import vispy.scene, vispy.app

from .graphics import TextureMaskFilter, ShadowRenderer
from .layers import GlyphRegistry, SpriteLayer
from .render_vispy import VispyLayerRenderer
from .maze import Maze
from .array_cache import ArraySumCache
from .entity import Entity


class Scene(Entity):
    """Central organizing class for managing UI, landscape, player, items, and mobs
    """
    def __init__(self, ui):
        Entity.__init__(self, entity_type='scene')
        self._player = None

        # game-owned render layers: the bridge between game state and renderers
        self.glyphs = GlyphRegistry()
        self.sprite_layers = {name: SpriteLayer(name) for name in ('scenery', 'items', 'actors')}

        # create maze
        self.maze = Maze.load_image('level1.png')

        # add scenery sprites for drawing maze
        self.maze.add_scenery(self.glyphs, self.sprite_layers['scenery'])

        # rendering backend consumes the layers
        self.renderer = VispyLayerRenderer(ui, self.glyphs, list(self.sprite_layers.values()))

        # line-of-sight computation
        opacity = self.maze.opacity.astype('float32')
        tr = self.renderer.txt.transforms.get_transform('framebuffer', 'visual')
        
        ms = self.maze.shape
        self.supersample = 4
        self.texture_shape = (ms[0] * self.supersample, ms[1] * self.supersample, 3)

        self.shadow_renderer = ShadowRenderer(self.maze, ui.canvas, supersample=self.supersample)
        self.norm_light = None

        self.light_cache = ArraySumCache()

        self.memory = np.zeros(self.texture_shape, dtype='float32')
        self.sight = np.zeros(self.texture_shape, dtype='float32')
        
        # filters scene for lighting, line of sight, and memory
        self.sight_texture =  vispy.gloo.Texture2D(shape=self.texture_shape, format='rgb', interpolation='linear', wrapping='repeat')
        self.sight_filter = TextureMaskFilter(self.sight_texture, tr, scale=(1./ms[1], 1./ms[0]))
        self.renderer.txt.attach(self.sight_filter)

        # track all items
        self.items = []
        
        # track monsters by location
        self.monsters = {}

        self._need_los_update = True

        ui.canvas.events.draw.connect(self.on_draw)
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

    def request_player_action(self, action):
        if action == 'take':
            items = self.items_at(self.player.location.slot)
            if len(items) == 0:
                self.console.write("Nothing to take here.")
            else:
                for item in items:
                    self.player.take(item)
                    self.console.write("Taken: %s" % item.name)
        elif action == 'read':
            self.player.read_item()

    def user_request_item(self, message, items, callback):
        """Ask the user to select an item from a list.
        """
        self.console.write(message)
        while True:
            ev = get_keypress()

    def on_draw(self, ev):
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
            # light = np.zeros(self.texture_shape, dtype='float32')
            light = self.light_cache.sum_arrays(lights)
            log_light = np.log(np.clip(light*10, 1, np.inf))
            self.norm_light = log_light / log_light.max()

        # current sight is combination of lighting and LOS
        self.sight = self.line_of_sight * self.norm_light

        self.sight_with_memory = self.memory * (1 - self.line_of_sight) + self.sight

        self.sight_texture.set_data(self.sight_with_memory.astype('float32'))

        # forget
        self.memory *= 0.999

        # add sight to memory
        self.memory[:, :, 2] = np.maximum(self.memory[:, :, 2], self.sight.max(axis=2))


