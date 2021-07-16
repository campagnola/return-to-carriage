# coding: utf8
import sys
import numpy as np
import vispy.scene, vispy.app
import vispy.util.ptime as ptime

from .graphics import CharAtlas, Sprites, TextureMaskFilter, CPUShadowRenderer, ShadowRenderer, Console
from .input import InputDispatcher, DefaultInputHandler, CommandInputHandler
from .player import Player
from .maze import Maze, MazeSprites
from .console import CommandInterpreter
from .array_cache import ArraySumCache


class Scene(object):
    """Central organizing class for managing UI, landscape, player, items, and mobs
    """
    def __init__(self, canvas):
        self.debug_line_of_sight = False
        self.debug_los_tex = False
        self.canvas = canvas
        
        # Setup input event handling
        self.input_dispatcher = InputDispatcher(canvas)
        self.default_input_handler = DefaultInputHandler(self)
        self.input_dispatcher.add_handler(self.default_input_handler)
        self.command_mode = False

        # setup UI
        self.view = canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.rect = [0, -5, 120, 60]
        self.view.camera.aspect = 0.6
        
        self.camera_target = self.view.camera.rect
        self._last_camera_update = ptime.time()
        self.scroll_timer = vispy.app.Timer(start=True, connect=self._scroll_camera, interval=0.016)        
        
        # generate a texture for each character we need
        self.atlas = CharAtlas()

        # create sprites visual
        self.txt = Sprites(self.atlas, sprite_size=(1, 1), point_cs='visual', parent=self.view.scene)

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

        self.shadow_renderer = ShadowRenderer(self, opacity, supersample=self.supersample)
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
        
        self.console_grid = self.canvas.central_widget.add_grid()

        self.stats_box = Console((2, 160))
        self.console_grid.add_widget(self.stats_box.view, 1, 0, 1, 2)
        self.stats_box.write("HP:17/33   Food:56%  Water:34%  Sleep:65%   Weight:207(45)    Level:3  Int:12  Str:9  Wis:11  Cha:2")
        self.stats_box.view.height_max = 30
        self.stats_box.view.stretch = (1, 10)
        

        self.info_box = Console((15, 80))
        self.console_grid.add_widget(self.info_box.view, 2, 0)
        self.info_box.write("There is a scroll of infinite recursion here.")
        self.info_box.view.height_max = 200
        self.stats_box.view.stretch = (1, 1)
        
        self.console = Console((15, 80))
        self.console_grid.add_widget(self.console.view, 2, 1)
        self.console.view.stretch = (1, 10)
        #self.console.view.parent = self.canvas.scene
        self.console.view.rect = vispy.geometry.Rect(30, 620, 1350, 250)
        self.console.transform = vispy.visuals.transforms.STTransform((0, 0, -0.5))
        #self.console.view.camera.aspect = 0.6
        
        self.console.view.height_max = 200

        self.console.write('Hello?')
        self.console.write('Is anybody\n    there?')
        self.console.write(''.join([chr(i) for i in range(0x20,128)]))        
        #self.console.view.camera.rect = [-1, -1, 30, 3]

        self.command = CommandInterpreter(self)
        self.cmd_input_handler = CommandInputHandler(self.console, self.command)

        self.canvas.events.draw.connect(self.on_draw)

    def toggle_command_mode(self):
        # todo: visual cue
        self.command_mode = not self.command_mode
        if self.command_mode:
            self.cmd_input_handler.activate()
        else:
            self.cmd_input_handler.deactivate()

    def monster_moved(self, mon, old_pos):
        if old_pos is not None:
            self.monsters[tuple(old_pos)].remove(mon)
        self.monsters.setdefault(tuple(mon.position), []).append(mon)
        
    def move_player(self, pos):
        self.player.location.update(self.maze, pos)
        self._need_los_update = True

        self._update_camera_target()

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

    def quit(self):
        self.canvas.close()

    def _update_camera_target(self):
        pp = np.array(self.player.location.global_location.slot)
        cr = vispy.geometry.Rect(self.view.camera.rect)
        cc = np.array(cr.center)
        cs = np.array(cr.size)
        cp = np.array(cr.pos)
        
        dif = pp - cc
        maxdif = 0.1 * cs  # start correcting camera at 10% width from center
        for ax in (0, 1):
            if dif[ax] <  -maxdif[ax]:
                cp[ax] += dif[ax] + maxdif[ax]
            elif dif[ax] > maxdif[ax]:
                cp[ax] += dif[ax] - maxdif[ax]
                
        cr.pos = cp
        self.camera_target = cr
        
    def _scroll_camera(self, ev):
        now = ptime.time()
        dt = now - self._last_camera_update
        self._last_camera_update = now
        
        cr = vispy.geometry.Rect(self.view.camera.rect)
        tr = self.camera_target
        
        crv = np.array(cr.pos + cr.size, dtype='float32')
        trv = np.array(tr.pos + tr.size, dtype='float32')
        
        if not np.any(abs(trv-crv) > 1e-2):
            return
        
        s = np.exp(-dt / 0.4)  # 400 ms settling time constant
        nrv = crv * s + trv * (1.0-s)
        
        cr.pos = nrv[:2]
        cr.size = nrv[2:]
        self.view.camera.rect = cr

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


