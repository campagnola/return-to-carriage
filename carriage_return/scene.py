# coding: utf8
import sys
import numpy as np
import vispy.scene, vispy.app

from .graphics import CharAtlas, Sprites, TextureMaskFilter, ShadowRenderer
from .player import Player
from .maze import Maze, MazeSprites
from .array_cache import ArraySumCache


class Scene(object):
    """Central organizing class for managing UI, landscape, player, items, and mobs
    """
    def __init__(self, ui):

        # generate a texture for each character we need
        self.atlas = CharAtlas()

        # create sprites visual
        self.txt = Sprites(self.atlas, sprite_size=(1, 1), point_cs='visual', parent=ui.view.scene)

        # create maze
        self.maze = Maze.load_image('level1.png')

        # add sprites for drawing maze
        self.maze.add_sprites(self.atlas, self.txt)

        # line-of-sight computation
        opacity = self.maze.opacity.astype('float32')
        tr = self.txt.transforms.get_transform('framebuffer', 'visual')
        
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
        self.txt.attach(self.sight_filter)

        # track all items
        self.items = []
        
        # track monsters by location
        self.monsters = {}

        # add player
        self.player = Player(self)

        self._need_los_update = True

        self.move_player([7, 7])

        ui.canvas.events.draw.connect(self.on_draw)
        self.player.location.global_changed.connect(ui._update_camera_target)

    def monster_moved(self, mon, old_pos):
        if old_pos is not None:
            self.monsters[tuple(old_pos)].remove(mon)
        self.monsters.setdefault(tuple(mon.position), []).append(mon)
        
    def move_player(self, pos):
        self.player.location.update(self.maze, pos)
        self._need_los_update = True

        self.end_turn()
        
    def end_turn(self):
        for mlist in list(self.monsters.values()):
            for m in mlist:
                m.take_turn()

    def add_item(self, item):
        self.items.append(item)

    def request_player_move(self, newpos):
        """Attempt to move the player to newpos.
        """
        pos = self.player.location.slot
        j, i = newpos
        j0, i0 = self.player.location.slot
        if self.maze.blocktype_at(i, j)['walkable']:
            self.move_player(newpos)
        elif self.maze.blocktype_at(i0, j)['walkable']:
            newpos[1] = i0
            self.move_player(newpos)
        elif self.maze.blocktype_at(i, j0)['walkable']:
            newpos[0] = j0
            self.move_player(newpos)
        self.norm_light = None

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


