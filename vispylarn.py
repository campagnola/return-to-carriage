# ~*~ coding: utf8 ~*~

import numpy as np
from PyQt4 import QtGui, QtCore
import vispy.visuals, vispy.scene, vispy.gloo


class SpritesVisual(vispy.visuals.Visual):
    vertex_shader = """
        #version 120

        uniform float size;

        attribute vec2 position;
        attribute vec4 fgcolor;
        attribute vec4 bgcolor;
        attribute float sprite;

        varying float v_sprite;
        varying vec4 v_fgcolor;
        varying vec4 v_bgcolor;

        void main (void) {
            v_fgcolor = fgcolor;
            v_bgcolor = bgcolor;
            v_sprite = sprite;

            gl_Position = $transform(vec4(position, 0, 1));
            gl_PointSize = size * 1.01;  // extra 0.01 prevents gaps
        }
    """

    fragment_shader = """
        #version 120
        uniform float size;
        varying vec4 v_fgcolor;
        varying vec4 v_bgcolor;
        varying float v_sprite;
        
        uniform sampler2D atlas;
        uniform sampler1D atlas_map;
        uniform float n_sprites;
        uniform vec2 scale;
        
        void main()
        {
            gl_FragColor = vec4(0, 0, 0, 0);
            vec4 atlas_coords = texture1D(atlas_map, (v_sprite + 0.5) / n_sprites);
            vec2 pt = gl_PointCoord.xy / scale;
            if( pt.x < 0 || pt.y < 0 || pt.x > 1 || pt.y > 1 ) {
                discard;
            }
            
            // supersample sprite value
            const int ss = 2;
            float alpha = 0;
            for (int i=0; i<ss; i++) {
                for (int j=0; j<ss; j++) {
                    vec2 dx = vec2(i/(size*ss), j/(size*ss));
                    vec2 tex_coords = atlas_coords.yx + (pt + dx/scale) * atlas_coords.wz;
                    vec4 tex = texture2D(atlas, tex_coords);
                    alpha += tex.g / (ss*ss);
                }
            }
            
            gl_FragColor = v_fgcolor * alpha + v_bgcolor * (1-alpha);
        }
    """

    def __init__(self, atlas, size=16, scale=1):
        self.size = size
        self.scale = scale
        self.atlas = atlas
        self.data = np.empty(1, dtype=[('pos', 'float32', 2), 
                                       ('sprite', 'uint32'), 
                                       ('fgcolor', 'float32', 4),
                                       ('bgcolor', 'float32', 4)])
        
        self._atlas_tex = vispy.gloo.Texture2D(shape=(1,1,4), format='rgba', interpolation='nearest')
        self._atlas_map_tex = vispy.gloo.Texture1D(shape=(1,4), format='rgba', interpolation='nearest')
        self._need_data_upload = False
        self._need_atlas_upload = False
        
        vispy.visuals.Visual.__init__(self, self.vertex_shader, self.fragment_shader)
        self._draw_mode = 'points'
    
    def add_sprites(self, n):
        """Expand to allow n more sprites, return a SpriteData instance.
        """
        i = self._resize(self.data.shape[0] + n)
        return SpriteData(self, i, i+n)

    def _resize(self, n):
        """Resize sprite array, return old size.
        """
        n1 = len(self.data)
        newdata = np.empty(n, dtype=self.data.dtype)
        newdata[:n1] = self.data[:n]
        self.data = newdata
        self._need_data_upload = True
        return n1
    
    def _upload_data(self):
        self.shared_program['position'] = self.data['pos']
        self.shared_program['sprite'] = self.data['sprite'].astype('float32')
        self.shared_program['fgcolor'] = self.data['fgcolor']
        self.shared_program['bgcolor'] = self.data['bgcolor']
        self.shared_program['size'] = self.size
        self.shared_program['scale'] = self.scale
        self._need_data_upload = False

    def _upload_atlas(self):
        self._atlas_tex.set_data(self.atlas.atlas)
        self.shared_program['atlas'] = self._atlas_tex
        self._atlas_map_tex.set_data(self.atlas.sprite_coords)
        self.shared_program['atlas_map'] = self._atlas_map_tex
        self.shared_program['n_sprites'] = self._atlas_map_tex.shape[0]
        self._need_atlas_upload = False
    
    def _prepare_transforms(self, view):
        xform = view.transforms.get_transform()
        view.view_program.vert['transform'] = xform
        
    def _prepare_draw(self, view):
        if self._need_data_upload:
            self._upload_data()
            
        if self._need_atlas_upload:
            self._upload_atlas()
        
        # set point size to match zoom
        tr = view.transforms.get_transform('visual', 'canvas')
        o = tr.map((0, 0))
        x = tr.map((self.size, 0))
        l = ((x-o)[:2]**2).sum()**0.5
        view.view_program['size'] = l

    def _compute_bounds(self, axis, view):
        return self.pos[:, axis].min(), self.pos[:, axis].max()

        
Sprites = vispy.scene.visuals.create_visual_node(SpritesVisual)


class SpriteData(object):
    def __init__(self, sprites, start, stop):
        self.sprites = sprites
        self.indices = (start, stop)
        
    @property
    def data(self):
        start, stop = self.indices
        return self.sprites.data[start:stop]



import pyqtgraph as pg


class CharAtlas(object):
    """Texture atlas containing rendered text characters.    
    """
    def __init__(self, size=128):
        self.size = size
        self.font = QtGui.QFont('monospace', self.size)
        self.chars = {}
        self._fm = QtGui.QFontMetrics(self.font)
        char_shape = (int(self._fm.height()), int(self._fm.maxWidth()))
        self.glyphs = np.empty((0,) + char_shape + (3,), dtype='ubyte')
        self._rebuild_atlas()

    def add_chars(self, chars):
        """Add new characters to the atlas, return the first new index.
        """
        oldn = self.glyphs.shape[0]
        newglyphs = np.empty((oldn + len(chars),) + self.glyphs.shape[1:], dtype=self.glyphs.dtype)
        newglyphs[:oldn] = self.glyphs
        self.glyphs = newglyphs
        
        char_shape = self.glyphs.shape[1:3]
        
        for i,char in enumerate(chars):
            self.chars[char] = oldn + i
            
            img = QtGui.QImage(char_shape[1], char_shape[0], QtGui.QImage.Format_RGB32)
            p = QtGui.QPainter()
            p.begin(img)
            brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
            p.fillRect(0, 0, char_shape[1], char_shape[0], brush)
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            p.setPen(pen)
            p.setFont(self.font)
            p.drawText(0, self._fm.ascent(), char)
            p.end()
            self.glyphs[oldn+i] = pg.imageToArray(img)[..., :3].transpose(1, 0, 2)
        
        self._rebuild_atlas()
        return oldn

    def _rebuild_atlas(self):
        gs = self.glyphs.shape
        self.atlas = self.glyphs.reshape((gs[0]*gs[1], gs[2], gs[3]))
        self.sprite_coords = np.empty((gs[0], 4), dtype='float32')
        self.sprite_coords[:,0] = np.arange(0, gs[0]*gs[1], gs[1])
        self.sprite_coords[:,1] = 0
        self.sprite_coords[:,2] = gs[1]
        self.sprite_coords[:,3] = gs[2]
        self.sprite_coords[:,::2] /= self.atlas.shape[0]
        self.sprite_coords[:,1::2] /= self.atlas.shape[1]



if __name__ == '__main__':
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    canvas.size = 1400,1000
    
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.rect = [0, 0, 20, 20]
    view.camera.aspect = 0.6
    
    # generate a texture for each character we need
    atlas = CharAtlas()
    atlas.add_chars(".#")
    
    # create sprites visual
    size = 1/0.6
    scale = (0.6, 1)
    txt = Sprites(atlas, size, scale, parent=view.scene)
    
    

    # create maze
    shape = (50, 120)
    npts = shape[0]*shape[1]
    maze = txt.add_sprites(npts)
    mazedata = maze.data.reshape(shape)
    
    # set wall/floor
    sprites = mazedata['sprite']
    sprites[:] = 1
    sprites[1:10, 1:10] = 0
    sprites[-10:-1, -10:-1] = 0
    sprites[20:39, 1:80] = 0
    sprites[5:30, 6] = 0
    sprites[35, 5:-5] = 0

    # set positions
    pos = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)# * (size * np.array(scale).reshape(1, 1, 2))
    mazedata['pos'] = pos.astype('float32')

    # set colors
    sprite_colors = np.array([
        [[0.2, 0.2, 0.2, 1.0], [0.0, 0.0, 0.0, 1.0]],  # path
        [[0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 1.0]],  # wall
    ], dtype='float32')
    color = sprite_colors[sprites]
    mazedata['fgcolor'] = color[...,0,:]
    mazedata['bgcolor'] = color[...,1,:]
    
    # randomize wall color a bit
    rock = np.random.normal(scale=0.01, size=shape + (1,))
    walls = sprites == 1
    n_walls = walls.sum()
    mazedata['bgcolor'][...,:3][walls] += rock[walls]

    # add player
    player_sprite = atlas.add_chars('&')
    player = txt.add_sprites(1)
    player.data['pos'] = (7, 7)
    player.data['sprite'] = player_sprite
    player.data['fgcolor'] = (0, 0, 0.3, 1)
    player.data['bgcolor'] = (0.5, 0.5, 0.5, 1)

    # add scroll
    scroll_sprite = atlas.add_chars(u'次')
    scroll = txt.add_sprites(1)
    scroll.data['pos'] = (5, 5)
    scroll.data['sprite'] = scroll_sprite
    scroll.data['fgcolor'] = (200, 0, 0, 1)
    scroll.data['bgcolor'] = (0, 0, 0, 1)
    

    txt._need_data_upload = True
    txt._need_atlas_upload = True
