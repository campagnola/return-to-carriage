# ~*~ coding: utf8 ~*~
import faulthandler
faulthandler.enable()

from PyQt4 import QtGui, QtCore
import vispy.scene, vispy.app
import numpy as np
from graphics import CharAtlas, Sprites, TextureMaskFilter, ShadowRenderer, Console
from PIL import Image
import vispy.util.ptime as ptime
from input import InputThread
import sys


class Scene(object):
    def __init__(self, canvas):
        self.debug_line_of_sight = False
        self.debug_los_tex = False
        self.canvas = canvas
        
        self.view = canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.rect = [0, -5, 120, 60]
        self.view.camera.aspect = 0.6
        canvas.events.key_press.connect(self.key_pressed)
        canvas.events.key_release.connect(self.key_released)
        
        self.camera_target = self.view.camera.rect
        self._last_camera_update = ptime.time()
        self.scroll_timer = vispy.app.Timer(start=True, connect=self._scroll_camera, interval=0.016)

        self.keys = set()
        self._last_input_update = ptime.time()
        self.input_timer = vispy.app.Timer(start=True, connect=self._handle_input, interval=0.016)
        
        try:
            self.input_thread = InputThread()
            self.input_thread.new_event.connect(self._gamepad_event)
        except Exception:
            self.input_thread = None
            sys.excepthook(*sys.exc_info())
        self._gamepad_state = {}
        
        # generate a texture for each character we need
        self.atlas = CharAtlas()
        self.atlas.add_chars(" .#")
        
        # create sprites visual
        size = 1/0.6
        scale = (0.6, 1)
        self.txt = Sprites(self.atlas, sprite_size=(1, 1), point_cs='visual', parent=self.view.scene)
        
        # create maze
        path = 1
        wall = 2
        maze = np.array(Image.open('level1.png'))[::-1,:,0]
        maze[maze>0] = wall
        maze[maze==0] = path
        #shape = (50, 120)
        #self.maze = np.empty(shape, dtype='uint32')
        #self.maze[:] = wall
        #self.maze[1:10, 1:10] = path
        #self.maze[25:35, 105:115] = path
        #self.maze[20:39, 1:80] = path
        #self.maze[5:30, 6] = path
        #self.maze[35, 5:115] = path
        #for i in range(24,36,10):
            #for j in range(15,70,15):
                #self.maze[i:i+2, j:j+2] = wall
        #self.maze[30, 3] = wall
        
        self.maze = maze
        shape = maze.shape
        self.path = path
        self.wall = wall
        
        self.maze_sprites = self.txt.add_sprites(shape)
        self.maze_sprites.sprite = self.maze

        # line-of-sight computation
        self.opacity = (self.maze == wall).astype('float32')
        #self.opacity_tex = vispy.gloo.Texture2D(self.opacity, format='luminance', interpolation='nearest')
        #self.sight_renderer = SightRenderer(self, self.opacity_tex)
        #self.sight_tex = vispy.gloo.Texture2D(shape=(1, 100), format='luminance', internalformat='r32f', interpolation='linear')
        ss = 1
        #self.los_tex_renderer = LOSTextureRenderer(self, self.sight_tex, self.maze.shape, supersample=ss)
        self.los_renderer = ShadowRenderer(self, self.opacity, supersample=ss)
        
        ms = self.maze.shape
        self.memory = np.zeros((ms[0]*ss, ms[1]*ss, 4), dtype='ubyte')
        self.memory[...,3] = 1
        self.memory_tex = vispy.gloo.Texture2D(self.memory, interpolation='linear')
        self.sight_filter = TextureMaskFilter(self.memory_tex, self.txt.shared_program['position'], self.maze.shape[:2][::-1])
        self.txt.attach(self.sight_filter)
        
        # set positions
        pos = np.zeros(shape + (3,), dtype='float32')
        pos[...,:2] = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)
        self.maze_sprites.position = pos

        # set colors
        sprite_colors = np.array([
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # nothing
            [[0.2, 0.2, 0.2, 1.0], [0.05, 0.05, 0.05, 1.0]],  # path
            [[0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 1.0]],  # wall
        ], dtype='float32')
        color = sprite_colors[self.maze]
        self.fgcolor = color[...,0,:]
        self.bgcolor = color[...,1,:]
        
        # randomize wall color a bit
        rock = np.random.normal(scale=0.01, size=shape + (1,))
        walls = self.maze == 2
        n_walls = walls.sum()
        self.bgcolor[...,:3][walls] += rock[walls]


        # add player
        self.player = Player(self)

        ## add items
        self.items = self.txt.add_sprites((10,))
        self.items.sprite = 0
        
        self.scroll = self.txt.add_sprites((1,))
        self.scroll.position = (5, 5, -0.1)
        self.scroll.sprite = self.atlas.add_chars(u'次')
        self.scroll.fgcolor = (0.7, 0, 0, 1)
        self.scroll.bgcolor = (0, 0, 0, 1)
        
        self.move_player([7, 7])
        self.update_sight()
        self.update_maze()
        
        self.console = Console()
        self.console.view.parent = self.canvas.scene
        self.console.view.rect = vispy.geometry.Rect(100, 100, 200, 200)
        
        
    def move_player(self, pos):
        self.player.position = pos
        self.update_sight()
        self.update_maze()
        #self.sight_filter.set_player_pos(pos)
        #img = self.sight_renderer.render(pos)

        #self.sight_tex.set_data(img.astype('float32'))
        #los = self.los_tex_renderer.render(pos)[::-1]
        los = self.los_renderer.render(pos)
        mask = np.where(los > self.memory, los, self.memory)
        self.memory[..., 2] = mask[..., 2]
        self.memory_tex.set_data(mask)

        self._update_camera_target()
        
        if self.debug_line_of_sight:
            if not hasattr(self, 'sight_plot'):
                import pyqtgraph as pg
                #self.sight_plot = pg.plot()
                #self.sight_plot.setYRange(0, 20)
                self.sight_img = pg.image()
                self.sight_img.imageItem.setBorder('w')
                #self.sight_img.resize(1200, 200)
                #self.sight_img.setLevels(0, 10)
            #theta = np.linspace(-np.pi, np.pi, img.shape[1])
            #self.sight_plot.plot(theta, img[img.shape[0]//2], clear=True)
            #self.sight_img.setImage(img.transpose(1, 0), autoLevels=False)
            self.sight_img.setImage(img.transpose(1, 0), autoLevels=False)


        if self.debug_los_tex:
            if not hasattr(self, 'los_tex_imv'):
                import pyqtgraph as pg
                self.los_tex_imv =  pg.image()
                self.los_tex_imv.imageItem.setBorder('w')
            self.los_tex_imv.setImage(los.transpose(1, 0, 2))
            
    def _update_camera_target(self):
        pp = np.array(self.player.position)
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

    def key_pressed(self, ev):
        if ev.key == 'Escape':
            self.canvas.close()
        
        self.keys.add(ev.key)
        self._handle_input(None)
        
    def key_released(self, ev):
        try:
            self.keys.remove(ev.key)
        except KeyError:
            pass
        
    def _gamepad_event(self, ev, state):
        # gamepad input
        self._gamepad_state = state
        self._handle_input(None)
 
    def _handle_input(self, ev):
        now = ptime.time()
        dt = now - self._last_input_update
        
        gp = self._gamepad_state
        gp_south = gp.get('BTN_SOUTH', 0) == 1
        wait = 0.05 if ('Shift' in self.keys or gp_south) else 0.1
        if dt < wait:
            return

        gp_x = gp.get('ABS_HAT0X', 0)
        gp_y = gp.get('ABS_HAT0Y', 0)
        dx = [gp_x, -gp_y]
        if 'Right' in self.keys:
            dx[0] += 1
        if 'Left' in self.keys:
            dx[0] -= 1
        if 'Up' in self.keys:
            dx[1] += 1
        if 'Down' in self.keys:
            dx[1] -= 1
        
        if dx[0] == 0 and dx[1] == 0:
            return
        
        pos = self.player.position
        j0, i0 = pos.astype('uint')
        newpos = pos + dx
        j, i = newpos.astype('uint')
        
        if self.maze[i, j] == self.path:
            self.move_player(newpos)
        elif self.maze[i0, j] == self.path:
            newpos[1] = i0
            self.move_player(newpos)
        elif self.maze[i, j0] == self.path:
            newpos[0] = j0
            self.move_player(newpos)
        
        self._last_input_update = now
 
    def update_sight(self):
        #self.sight_filter.set_player_pos(self.player.position[:2])
        pass
        
        
    def update_maze(self):
        mem = np.clip(self.memory[...,None], 0, 1)
        self.maze_sprites.fgcolor = self.fgcolor# * mem
        self.maze_sprites.bgcolor = self.bgcolor# * mem



class Player(object):
    def __init__(self, scene):
        self.scene = scene
        self.zval = -0.2
        self.sprite = self.scene.txt.add_sprites((1,))
        self.sprite.sprite = self.scene.atlas.add_chars('&')
        self.sprite.fgcolor = (0, 0, 0.3, 1)
        self.sprite.bgcolor = (0.5, 0.5, 0.5, 1)
        self.position = (7, 7)

    @property
    def position(self):
        return np.array(self._pos)
    
    @position.setter
    def position(self, p):
        self._pos = p
        self.sprite.position = tuple(p) + (self.zval,)



if __name__ == '__main__':
    import user
    
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    canvas.size = 1400,900
    
    scene = Scene(canvas)

    
    