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
        
        void main()
        {
            vec4 atlas_coords = texture1D(atlas_map, v_sprite);
            vec2 uv = gl_PointCoord.xy; // 0-1 over point area
            vec2 tex_coords = atlas_coords.xy + uv * atlas_coords.zw;
            vec4 tex = texture2D(atlas, tex_coords / atlas_size);
            gl_FragColor = tex.r * v_fgcolor + tex.g * v_bgcolor;
            gl_FragColor.a = 1.0;
            gl_FragColor.r = 1.0;
        }
    """

    def __init__(self):
        vispy.visuals.Visual.__init__(self, self.vertex_shader, self.fragment_shader)
        
        self.pos = np.random.normal(size=(10, 2), loc=300, scale=100).astype('float32')
        self.sprites = np.random.randint(0, 5, size=10).astype('float32')
        
        self.atlas = np.random.randint(0, 256, size=(10, 50, 3)).astype('ubyte')
        self.atlas_map = np.zeros((5, 4), dtype='float32')
        self.atlas_map[:,0] = np.arange(0, 50, 10)
        self.atlas_map[:,2:] = 10
        
        self.size = 40
        self.fgcolor = np.random.normal(size=(10, 4)).astype('float32')
        self.bgcolor = np.random.normal(size=(10, 4)).astype('float32')

        self.shared_program['position'] = self.pos
        self.shared_program['sprite'] = self.sprites
        self.shared_program['fgcolor'] = self.fgcolor
        self.shared_program['bgcolor'] = self.bgcolor
        self.shared_program['size'] = self.size
        
        self._atlas_tex = vispy.gloo.Texture2D(data=self.atlas, interpolation='nearest')
        self.shared_program['atlas'] = self._atlas_tex

        self.shared_program['atlas_size'] = self._atlas_tex.shape[:2]

        self._atlas_map_tex = vispy.gloo.Texture1D(data=self.atlas_map, interpolation='nearest')
        self.shared_program['atlas_map'] = self._atlas_map_tex

        self._draw_mode = 'points'
        
    def _prepare_transforms(self, view):
        xform = view.transforms.get_transform()
        view.view_program.vert['transform'] = xform

    def _prepare_draw(self, view):
        pass

    def _compute_bounds(self, axis, view):
        return self.pos[:, axis].min(), self.pos[:, axis].max()

        
Sprites = vispy.scene.visuals.create_visual_node(SpritesVisual)


if __name__ == '__main__':
    canvas = vispy.scene.SceneCanvas()
    canvas.show()
    
    #view = canvas.central_widget.add_view()
    #view.camera = 'panzoom'
    #view.camera.rect = [0, 0, 200, 200]
    
    txt = Sprites(parent=canvas.scene)