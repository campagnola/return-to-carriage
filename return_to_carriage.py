# ~*~ coding: utf8 ~*~
from PyQt4 import QtGui, QtCore
import vispy.scene
import numpy as np
from graphics import CharAtlas, Sprites



class Scene(object):
    def __init__(self, view):
        
        # generate a texture for each character we need
        self.atlas = CharAtlas()
        self.atlas.add_chars(".#")
        
        # create sprites visual
        size = 1/0.6
        scale = (0.6, 1)
        self.txt = Sprites(self.atlas, size, scale, parent=view.scene)
        
        # create maze
        shape = (50, 120)
        self.maze = np.ones(shape, dtype='uint32')
        self.maze[1:10, 1:10] = 0
        self.maze[25:35, 105:115] = 0
        self.maze[20:39, 1:80] = 0
        self.maze[5:30, 6] = 0
        self.maze[35, 5:115] = 0
        
        self.maze_sprites = self.txt.add_sprites(shape)
        self.maze_sprites.sprite = self.maze

        # set positions
        pos = np.zeros(shape + (3,), dtype='float32')
        pos[...,:2] = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)
        self.maze_sprites.position = pos

        # set colors
        sprite_colors = np.array([
            [[0.2, 0.2, 0.2, 1.0], [0.0, 0.0, 0.0, 1.0]],  # path
            [[0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 1.0]],  # wall
        ], dtype='float32')
        color = sprite_colors[self.maze]
        self.maze_sprites.fgcolor = color[...,0,:]
        bgcolor = color[...,1,:]
        
        # randomize wall color a bit
        rock = np.random.normal(scale=0.01, size=shape + (1,))
        walls = self.maze == 1
        n_walls = walls.sum()
        bgcolor[...,:3][walls] += rock[walls]
        self.maze_sprites.bgcolor = bgcolor

        # add player
        self.player = self.txt.add_sprites((1,))
        self.player.position = (7, 7, -0.2)
        self.player.sprite = self.atlas.add_chars('&')
        self.player.fgcolor = (0, 0, 0.3, 1)
        self.player.bgcolor = (0.5, 0.5, 0.5, 1)

        ## add scroll
        self.scroll = self.txt.add_sprites((1,))
        self.scroll.position = (5, 5, -0.1)
        self.scroll.sprite = self.atlas.add_chars(u'次')
        self.scroll.fgcolor = (0.7, 0, 0, 1)
        self.scroll.bgcolor = (0, 0, 0, 1)
        
    def key_pressed(self, ev):
        pos = self.player.position
        if ev.key == 'Right':
            dx = (1, 0, 0)
        elif ev.key == 'Left':
            dx = (-1, 0, 0)
        elif ev.key == 'Up':
            dx = (0, 1, 0)
        elif ev.key == 'Down':
            dx = (0, -1, 0)
        else:
            return
        
        newpos = pos + dx
        j, i = tuple(newpos[0,:2].astype('uint'))
        if self.maze[i, j] == 0:
            self.player.position = newpos
            
        


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

    