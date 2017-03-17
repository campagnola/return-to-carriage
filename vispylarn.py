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
            gl_PointSize = size;
        }
    """

    fragment_shader = """
        #version 120
        varying vec4 v_fgcolor;
        varying vec4 v_bgcolor;
        varying float v_sprite;
        
        uniform sampler2D atlas;
        uniform vec2 atlas_size;
        uniform sampler1D atlas_map;
        uniform float n_sprites;
        uniform vec2 scale;
        
        void main()
        {
            vec4 atlas_coords = texture1D(atlas_map, v_sprite / n_sprites);
            vec2 pt = gl_PointCoord.xy / scale;
            if( pt.x < 0 || pt.y < 0 || pt.x > 1 || pt.y > 1 ) {
                discard;
            }
            vec2 tex_coords = atlas_coords.yx + pt * atlas_coords.wz;
            vec4 tex = texture2D(atlas, tex_coords);
            gl_FragColor = tex.g * v_fgcolor + tex.b * v_bgcolor;
            gl_FragColor.a = 1.0;
            gl_FragColor.rgb += 0.1;
        }
    """

    def __init__(self, shape=(40, 120)):
        vispy.visuals.Visual.__init__(self, self.vertex_shader, self.fragment_shader)
        
        self.size = 16
        scale = (0.6, 1)
        npts = shape[0]*shape[1]
        self.pos = np.mgrid[0:shape[1], 0:shape[0]].transpose(1, 2, 0).reshape(npts,2) * (self.size * np.array(scale).reshape(1, 1, 2))
        self.pos = self.pos.astype('float32')
        self.sprites = np.random.randint(0, 5, size=shape).astype('float32').reshape(npts)
        
        self.atlas = SpriteAtlas()
        
        self.fgcolor = np.random.normal(size=(npts, 4)).astype('float32')
        self.bgcolor = np.random.normal(size=(npts, 4)).astype('float32')

        self.shared_program['position'] = self.pos
        self.shared_program['sprite'] = self.sprites
        self.shared_program['fgcolor'] = self.fgcolor
        self.shared_program['bgcolor'] = self.bgcolor
        self.shared_program['size'] = self.size
        self.shared_program['scale'] = scale
        
        self._atlas_tex = vispy.gloo.Texture2D(data=self.atlas.atlas, interpolation='nearest')
        self.shared_program['atlas'] = self._atlas_tex

        self.shared_program['atlas_size'] = self._atlas_tex.shape[:2]

        self._atlas_map_tex = vispy.gloo.Texture1D(data=self.atlas.sprite_coords, interpolation='nearest')
        self.shared_program['atlas_map'] = self._atlas_map_tex

        self.shared_program['n_sprites'] = self._atlas_map_tex.shape[0]


        self._draw_mode = 'points'
        
    def _prepare_transforms(self, view):
        xform = view.transforms.get_transform()
        view.view_program.vert['transform'] = xform

    def _prepare_draw(self, view):
        pass

    def _compute_bounds(self, axis, view):
        return self.pos[:, axis].min(), self.pos[:, axis].max()

        
Sprites = vispy.scene.visuals.create_visual_node(SpritesVisual)

import pyqtgraph as pg
class SpriteAtlas(object):
    def __init__(self):
        chars = "12345"
        self.size = 32
        self.font = QtGui.QFont('monospace', self.size)
        fm = QtGui.QFontMetrics(self.font)
        char_shape = (int(fm.height()), int(fm.maxWidth()))
        self.glyphs = np.empty((len(chars),) + char_shape + (3,), dtype='ubyte')
        for i,char in enumerate(chars):
            img = QtGui.QImage(char_shape[1], char_shape[0], QtGui.QImage.Format_RGB32)
            p = QtGui.QPainter()
            p.begin(img)
            brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
            p.fillRect(0, 0, char_shape[1], char_shape[0], brush)
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            p.setPen(pen)
            p.setFont(self.font)
            p.drawText(0, fm.ascent(), char)
            p.end()
            self.glyphs[i] = pg.imageToArray(img)[..., :3].transpose(1, 0, 2)
            
        gs = self.glyphs.shape
        self.atlas = self.glyphs.reshape((gs[0]*gs[1], gs[2], gs[3]))
        self.sprite_coords = np.empty((gs[0], 4), dtype='float32')
        self.sprite_coords[:,0] = np.arange(0, gs[0]*gs[1], gs[1])
        self.sprite_coords[:,1] = 0
        self.sprite_coords[:,2] = char_shape[0]
        self.sprite_coords[:,3] = char_shape[1]
        
        self.sprite_coords[:,::2] /= self.atlas.shape[0]
        self.sprite_coords[:,1::2] /= self.atlas.shape[1]


if __name__ == '__main__':
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    canvas.size = 1400,1000
    
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.rect = [0, 0, 200, 200]
    
    txt = Sprites(parent=view.scene)
