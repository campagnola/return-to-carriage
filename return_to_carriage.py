# ~*~ coding: utf8 ~*~
from PyQt4 import QtGui, QtCore
import vispy.scene
import numpy as np
from graphics import CharAtlas, Sprites


class Scene(object):
    def __init__(self, view):
        
        # generate a texture for each character we need
        self.atlas = CharAtlas()
        self.atlas.add_chars(" .#")
        
        # create sprites visual
        size = 1/0.6
        scale = (0.6, 1)
        self.txt = Sprites(self.atlas, size, scale, parent=view.scene)
        
        # create maze
        shape = (50, 120)
        path = 1
        wall = 2
        self.maze = np.empty(shape, dtype='uint32')
        self.maze[:] = wall
        self.maze[1:10, 1:10] = path
        self.maze[25:35, 105:115] = path
        self.maze[20:39, 1:80] = path
        self.maze[5:30, 6] = path
        self.maze[35, 5:115] = path
        self.path = path
        self.wall = wall
        
        self.memory = np.zeros(shape, dtype='float32')
        self.sight = np.zeros(shape, dtype='float32')
        
        self.maze_sprites = self.txt.add_sprites(shape)
        self.maze_sprites.sprite = self.maze

        # set positions
        pos = np.zeros(shape + (3,), dtype='float32')
        pos[...,:2] = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)
        self.maze_sprites.position = pos

        # set colors
        sprite_colors = np.array([
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],  # nothing
            [[0.2, 0.2, 0.2, 1.0], [0.0, 0.0, 0.0, 1.0]],  # path
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
        
        self.update_sight()
        self.update_maze()
        
    def key_pressed(self, ev):
        pos = self.player.position
        if ev.key == 'Right':
            dx = (1, 0)
        elif ev.key == 'Left':
            dx = (-1, 0)
        elif ev.key == 'Up':
            dx = (0, 1)
        elif ev.key == 'Down':
            dx = (0, -1)
        else:
            return
        
        newpos = pos + dx
        j, i = newpos.astype('uint')
        if self.maze[i, j] == self.path:
            self.player.position = newpos
            self.update_sight()
            self.update_maze()
            
    def update_sight(self):
        self.sight[:] = 0
        pos = self.player.position[:2].astype(int)
        bl = np.clip(pos-3, 0, self.maze.size)
        tr = pos+4
        self.sight[bl[1]:tr[1], bl[0]:tr[0]] = 1
        
        self.memory *= 0.998
        #self.memory = np.where(self.memory > self.sight, self.memory, self.sight)
        self.memory += self.sight
        
    def update_maze(self):
        mem = np.clip(self.memory[...,None], 0, 1)
        self.maze_sprites.fgcolor = self.fgcolor * mem
        self.maze_sprites.bgcolor = self.bgcolor * mem



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
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    canvas.size = 1400,900
    
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.rect = [0, -5, 120, 60]
    view.camera.aspect = 0.6
    
    scene = Scene(view)
    canvas.events.key_press.connect(scene.key_pressed)

    