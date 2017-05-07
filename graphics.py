
import numpy as np
from PyQt4 import QtGui, QtCore
import vispy.visuals, vispy.scene, vispy.gloo
from vispy.visuals.shaders import ModularProgram, Function, Varying
from vispy.visuals.transforms import STTransform, PolarTransform, NullTransform


# load support for opengl 3 features
vispy.gloo.gl.use_gl('gl+')


class SpritesVisual(vispy.visuals.Visual):
    """
    2 basic ways we might want to draw sprites:
    
        - Size / anchor offset measured in visual coordinates
        - Size / anchor offset measured in canvas coordinates
    
    And 2 methods:
    
        - GLSL 120 drawing point_sprites
            map to sprite cs, set corners -> map to canvas, set point size -> map to render
        - GLSL 150+ with geometry shader converting points into quads
    
    
    
    """
    vertex_shader_2 = """
        #version 120

        attribute vec3 position;
        attribute float sprite;
        attribute vec4 fgcolor;
        attribute vec4 bgcolor;

        varying float v_sprite;
        varying vec4 v_fgcolor;
        varying vec4 v_bgcolor;
        varying vec2 v_point_coord_scale;
        varying vec2 v_point_size;

        void main (void) {
            v_fgcolor = fgcolor;
            v_bgcolor = bgcolor;
            v_sprite = sprite;

            // Map sprite location to the coordinate system where the size of
            // the sprite is specified
            vec4 anchor_pos = $visual_to_sprite(vec4(position, 1));
            anchor_pos /= anchor_pos.w;
            
            // Find corners of sprite and map to pixel coordinates
            vec4 bl = $sprite_to_px(anchor_pos - vec4($sprite_size, 0, 0) / 2.);
            vec4 tr = $sprite_to_px(anchor_pos + vec4($sprite_size, 0, 0) / 2.);
            bl /= bl.w;
            tr /= tr.w;
            
            // Measure pixel size of sprite
            vec2 size = abs(tr.xy - bl.xy);
            float ps = max(size.x, size.y) * 1.05;  // extra 0.01 prevents gaps
            gl_PointSize = ps;
            
            // points must be square; give scale factors to the fragment shader
            // to allow rectangular clipping.
            v_point_coord_scale = ps / size;
            v_point_size = size;
            
            // Map center of sprite to render coordinates
            vec4 center = (bl + tr) / 2.0;
            gl_Position = $px_to_render(center);
        }
    """

    fragment_shader_2 = """
        #version 120
        varying vec4 v_fgcolor;
        varying vec4 v_bgcolor;
        varying float v_sprite;
        varying vec2 v_point_coord_scale;
        varying vec2 v_point_size;
        
        uniform sampler2D atlas;
        uniform sampler1D atlas_map;
        uniform float n_sprites;
        
        void main()
        {
            gl_FragColor = vec4(0, 0, 0, 0);
            vec4 atlas_coords = texture1D(atlas_map, (v_sprite + 0.5) / n_sprites);
            vec2 pt = gl_PointCoord.xy * v_point_coord_scale;
            if( pt.x < 0 || pt.y < 0 || pt.x > 1 || pt.y > 1 ) {
                discard;
            }
            
            // supersample sprite value
            const float ss = 4;
            float alpha = 0;
            for (int i=0; i<ss; i++) {
                for (int j=0; j<ss; j++) {
                    vec2 dx = vec2(i/ss, j/ss) / v_point_size;
                    vec2 tex_coords = atlas_coords.yx + (pt + dx) * atlas_coords.wz;
                    vec4 tex = texture2D(atlas, tex_coords);
                    alpha += tex.g / (ss*ss);
                }
            }
            
            gl_FragColor = v_fgcolor * alpha + v_bgcolor * (1-alpha);
        }
    """

    vertex_shader_3 = """
        #version 330 compatibility

        in vec3 position;
        in float sprite;
        in vec4 fgcolor;
        in vec4 bgcolor;

        out float v_sprite;
        out vec4 v_fgcolor;
        out vec4 v_bgcolor;
        out vec2 v_sprite_size;

        void main (void) {
            v_fgcolor = fgcolor;
            v_bgcolor = bgcolor;
            v_sprite = sprite;
            v_sprite_size = $sprite_size;

            gl_Position = vec4(position, 1);
        }
    """

    geometry_shader_3 = """
        #version 330 compatibility
        
        layout (points) in;
        layout (triangle_strip, max_vertices=4) out;
        
        in float v_sprite[];
        in vec4 v_fgcolor[];
        in vec4 v_bgcolor[];
        in vec2 v_sprite_size[];

        out float f_sprite;
        out vec4 f_fgcolor;
        out vec4 f_bgcolor;
        out vec2 point_coord;
        out vec2 point_size;
        
        uniform vec2 scale;

        void main(void) {
            f_sprite = v_sprite[0];
            f_fgcolor = v_fgcolor[0];
            f_bgcolor = v_bgcolor[0];
        
            // Map sprite location to the coordinate system where the size of
            // the sprite is specified
            vec4 anchor_pos = $visual_to_sprite(gl_in[0].gl_Position);
            anchor_pos /= anchor_pos.w;
            
            // Find corners of sprite and map to pixel coordinates
            vec4 ss_half = vec4(v_sprite_size[0], 0, 0) / 2.;
            vec4 bl = $sprite_to_px(anchor_pos - ss_half);
            vec4 tr = $sprite_to_px(anchor_pos + ss_half);
            ss_half.x *= -1;
            vec4 br = $sprite_to_px(anchor_pos - ss_half);
            vec4 tl = $sprite_to_px(anchor_pos + ss_half);
            bl /= bl.w;
            tr /= tr.w;
            br /= br.w;
            tl /= tl.w;
            
            // Measure pixel size of sprite; fragment shader uses this to
            // supersample the sprite texture
            point_size = vec2(length(tr - tl), length(tr - br));
            
            // Map corners of sprite to render coordinates
            gl_Position = $px_to_render(bl);
            point_coord = vec2(0, 1);
            EmitVertex();
            gl_Position = $px_to_render(br);
            point_coord = vec2(1, 1);
            EmitVertex();
            gl_Position = $px_to_render(tl);
            point_coord = vec2(0, 0);
            EmitVertex();
            gl_Position = $px_to_render(tr);
            point_coord = vec2(1, 0);
            EmitVertex();
            EndPrimitive();
        }
    """

    fragment_shader_3 = """
        #version 330 compatibility
        
        uniform float size;
        in vec4 f_fgcolor;
        in vec4 f_bgcolor;
        in float f_sprite;
        in vec2 point_coord;
        in vec2 point_size;
        
        uniform sampler2D atlas;
        uniform sampler1D atlas_map;
        uniform float n_sprites;
        
        void main()
        {
            gl_FragColor = vec4(0, 0, 0, 0);
            vec4 atlas_coords = texture1D(atlas_map, (f_sprite + 0.5) / n_sprites);
            vec2 pt = point_coord;
            vec2 tex_coords;
            
            // supersample sprite value
            const int ss = 1;
            float alpha = 0;
            for (int i=0; i<ss; i++) {
                for (int j=0; j<ss; j++) {
                    vec2 dx = vec2(0, 0); //vec2(i, j) / (point_size*ss);
                    tex_coords = vec2(0.00001, 0.00001) + atlas_coords.yx + (pt + dx) * atlas_coords.wz;
                    vec4 tex = texture2D(atlas, tex_coords);
                    alpha += tex.g / (ss*ss);
                }
            }
            
            gl_FragColor = f_fgcolor * alpha + f_bgcolor * (1-alpha);
        }
    """
    
    def __init__(self, atlas, sprite_size=(16, 16), point_cs='pixel', method=None):
        if method is None:
            if 'GL_GEOMETRY_SHADER' in vispy.gloo.gl.__dict__:
                method = 'geometry'
            else:
                method = 'point_sprite'

        self.method = method
        self.point_cs = point_cs
        
        self.atlas = atlas
        self.atlas.atlas_changed.connect(self._atlas_changed)
        self.position = np.empty((0, 3), dtype='float32')
        self.sprite = np.empty((0,), dtype='uint32')
        self.sprite_size = sprite_size
        self.fgcolor = np.empty((0, 4), dtype='float32')
        self.bgcolor = np.empty((0, 4), dtype='float32')
        
        self._atlas_tex = vispy.gloo.Texture2D(shape=(1,1,4), format='rgba', interpolation='nearest')
        self._atlas_map_tex = vispy.gloo.Texture1D(shape=(1,4), format='rgba', internalformat='rgba32f', interpolation='nearest')
        self._need_data_upload = False
        self._need_atlas_upload = True
        
        self._null_transform = NullTransform()
        
        if method == 'point_sprite':
            shaders = self.vertex_shader_2, self.fragment_shader_2
        elif method == 'geometry':
            shaders = self.vertex_shader_3, self.fragment_shader_3, self.geometry_shader_3
        else:
            raise ValueError('method must be "point_sprite" or "geometry"')
        vispy.visuals.Visual.__init__(self, *shaders)

        self._draw_mode = 'points'
        self.shared_program['position'] = vispy.gloo.VertexBuffer()
        self.shared_program['sprite'] = vispy.gloo.VertexBuffer()
        self.shared_program['fgcolor'] = vispy.gloo.VertexBuffer()
        self.shared_program['bgcolor'] = vispy.gloo.VertexBuffer()
        
        self.update_gl_state(depth_test=True)
    
    def add_sprites(self, shape):
        """Expand to allow more sprites, return a SpriteData instance with the specified shape.
        """
        if not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple")
        n = np.product(shape)
        i = self._resize(self.position.shape[0] + n)
        return SpriteData(self, i, shape)

    def _resize(self, n):
        """Resize sprite array, return old size.
        """
        n1 = self.position.shape[0]        
        self.position = np.resize(self.position, (n, 3))
        self.sprite = np.resize(self.sprite, (n,))
        self.fgcolor = np.resize(self.fgcolor, (n, 4))
        self.bgcolor = np.resize(self.bgcolor, (n, 4))        
        self._upload_data()
        return n1
    
    def _upload_data(self):
        self.shared_program['position'].set_data(self.position)
        self.shared_program['sprite'].set_data(self.sprite.astype('float32'))
        self.shared_program['fgcolor'].set_data(self.fgcolor)
        self.shared_program['bgcolor'].set_data(self.bgcolor)
        self.shared_program.vert['sprite_size'] = tuple(self.sprite_size)
            
        self._need_data_upload = False

    def _atlas_changed(self, ev):
        self._need_atlas_upload = True
        self.update()

    def _upload_atlas(self):
        self._atlas_tex.set_data(self.atlas.atlas)
        self.shared_program['atlas'] = self._atlas_tex
        self._atlas_map_tex.set_data(self.atlas.sprite_coords)
        self.shared_program['atlas_map'] = self._atlas_map_tex
        self.shared_program['n_sprites'] = self._atlas_map_tex.shape[0]
        self._need_atlas_upload = False
    
    def _prepare_transforms(self, view):
        trs = view.transforms
        if self.point_cs == 'pixel':
            vis_to_sprite = trs.get_transform('visual', 'document').simplified
            sprite_to_px = self._null_transform
            px_to_render = trs.get_transform('document', 'render').simplified
        elif self.point_cs == 'visual':
            vis_to_sprite = self._null_transform
            sprite_to_px = trs.get_transform('visual', 'document').simplified
            px_to_render = trs.get_transform('document', 'render').simplified
        else:
            raise ValueError('Invalid point coordinate system "%s"' % self.point_cs)
        
        if self.method == 'point_sprite':
            view.view_program.vert['visual_to_sprite'] = vis_to_sprite
            view.view_program.vert['sprite_to_px'] = sprite_to_px
            view.view_program.vert['px_to_render'] = px_to_render
        elif self.method == 'geometry':
            view.view_program.geom['visual_to_sprite'] = vis_to_sprite
            view.view_program.geom['sprite_to_px'] = sprite_to_px
            view.view_program.geom['px_to_render'] = px_to_render
        else:
            raise ValueError('Invalid draw method "%s"' % self.method)
        
    def _prepare_draw(self, view):
        if self._need_data_upload:
            self._upload_data()
            
        if self._need_atlas_upload:
            self._upload_atlas()
        
        # set point size to match zoom
        #tr = view.transforms.get_transform('visual', 'canvas')
        #o = tr.map((0, 0))
        #x = tr.map((self.size, 0))
        #l = ((x-o)[:2]**2).sum()**0.5
        view.view_program.vert['sprite_size'] = tuple(self.sprite_size)

    def _compute_bounds(self, axis, view):
        p = self.position[:, axis]
        return p.min(), p.max()

        
Sprites = vispy.scene.visuals.create_visual_node(SpritesVisual)


class SpriteData(object):
    """For accessing a subset of sprites from a Sprites visual.
    
    This is intended to be created using SpritesVisual.add_sprites(...)
    """
    def __init__(self, sprites, start, shape):
        self.sprites = sprites
        n = np.product(shape)
        self.indices = (start, start+n)
        self.shape = shape
        
    @property
    def position(self):
        start, stop = self.indices
        return self.sprites.position[start:stop].reshape(self.shape + (3,))
    
    @position.setter
    def position(self, p):
        start, stop = self.indices
        self.position[:] = p
        self.sprites.shared_program['position'][start:stop] = self.position.view(dtype=[('position', 'float32', 3)]).reshape(stop-start)
        self.sprites.update()
        
    @property
    def sprite(self):
        start, stop = self.indices
        return self.sprites.sprite[start:stop].reshape(self.shape)
    
    @sprite.setter
    def sprite(self, p):
        start, stop = self.indices
        self.sprite[:] = p
        self.sprites.shared_program['sprite'][start:stop] = self.sprite.reshape(stop-start).astype('float32')
        self.sprites.update()

    @property
    def fgcolor(self):
        start, stop = self.indices
        return self.sprites.fgcolor[start:stop].reshape(self.shape + (4,))
    
    @fgcolor.setter
    def fgcolor(self, p):
        start, stop = self.indices
        self.fgcolor[:] = p
        self.sprites.shared_program['fgcolor'][start:stop] = self.fgcolor.view(dtype=[('fgcolor', 'float32', 4)]).reshape(stop-start)
        self.sprites.update()
        
    @property
    def bgcolor(self):
        start, stop = self.indices
        return self.sprites.bgcolor[start:stop].reshape(self.shape + (4,))
    
    @bgcolor.setter
    def bgcolor(self, p):
        start, stop = self.indices
        self.bgcolor[:] = p
        self.sprites.shared_program['bgcolor'][start:stop] = self.bgcolor.view(dtype=[('bgcolor', 'float32', 4)]).reshape(stop-start)
        self.sprites.update()
        


import pyqtgraph as pg


class CharAtlas(object):
    """Texture atlas containing rendered text characters.    
    """
    def __init__(self, size=128):
        self.atlas_changed = vispy.util.event.EventEmitter(type='atlas_changed')
        self.size = size
        self.columns = 8000 // size
        self.font = QtGui.QFont('monospace', self.size)
        self.chars = {}
        self._fm = QtGui.QFontMetrics(self.font)
        char_shape = (int(self._fm.height()), int(self._fm.width('x')))
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
        self.atlas_changed()
        return oldn

    def _rebuild_atlas(self):
        gs = self.glyphs.shape
        n_glyphs = gs[0]
        atlas_rows = int(np.ceil(float(n_glyphs) / self.columns))
        if atlas_rows == 1:
            columns = n_glyphs
        else:
            columns = self.columns
            
        self.atlas = np.empty((atlas_rows * gs[1], columns*gs[2], 3), dtype=self.glyphs.dtype)
        
        self.sprite_coords = np.empty((n_glyphs, 4), dtype='float32')
        self.sprite_coords[:,2] = gs[1]
        self.sprite_coords[:,3] = gs[2]
        for i in range(atlas_rows):
            for j in range(columns):
                glyph = i * columns + j
                if glyph >= n_glyphs:
                    break
                x = i * gs[1]
                y = j * gs[2]
                self.sprite_coords[glyph, 0:2] = x, y
                self.atlas[x:x+gs[1], y:y+gs[2]] = self.glyphs[glyph]
        
        self.sprite_coords[:,::2] /= self.atlas.shape[0]
        self.sprite_coords[:,1::2] /= self.atlas.shape[1]
        

class TextureMaskFilter(object):
    def __init__(self, texture, transform, scale):
        self.fshader = Function("""
            void apply_texture_mask() {
                vec4 tex_pos = $transform(gl_FragCoord);
                tex_pos /= tex_pos.w;
                vec4 mask = texture2D($texture, tex_pos.xy);
                gl_FragColor = gl_FragColor * mask;
            }
        """)
        self.fshader['texture'] = texture
        self.scale_tr = STTransform(scale=scale) * STTransform(translate=(0.5, 0.5))
        self.fshader['transform'] = self.scale_tr * transform
        
    def _attach(self, visual):
        self._visual = visual        
        fhook = visual._get_hook('frag', 'post')
        fhook.add(self.fshader(), position=3)


class LineOfSightFilter1(object):
    """Measure line=of-sight for each vertex as it is drawn.
    """
    def __init__(self, opacity, pos):
        self.opacity = opacity
        self.vshader = Function("""
            void line_of_sight() {
                vec2 diff = $player_pos.xy - $pos.xy;
                float dist = length(diff);
                float s = 0.5;
                vec2 step = s * diff / dist;
                int steps = int(dist/s);
                $visible = 1;
                vec2 sample_pos = $pos.xy + step/s;
                for( int i=1; i<steps; i++ ) {
                    float op = texture2D($opacity, (sample_pos+0.5) / $opacity_size).r;
                    if( op > 0 ) {
                        $visible = 0;
                        break;
                    }
                    sample_pos += step;
                }
            }
        """)
        
        self.fshader = Function("""
            void line_of_sight() {
                gl_FragColor = gl_FragColor * $visible;
                //gl_FragColor.r = $visible;
            }
        """)
        
        self.vshader['opacity'] = opacity
        self.vshader['opacity_size'] = opacity.shape[:2][::-1]
        self.vshader['pos'] = pos
        self.vshader['visible'] = Varying('visible', dtype='float')
        self.fshader['visible'] = self.vshader['visible']
        
    def set_player_pos(self, pos):
        self.vshader['player_pos'] = pos

    def _attach(self, visual):
        self._visual = visual
        fhook = visual._get_hook('frag', 'post')
        fhook.add(self.fshader(), position=3)
        vhook = visual._get_hook('vert', 'post')
        vhook.add(self.vshader(), position=3)

        #self.vshader['position'] = visual.shared_program.vert['position']


class LineOfSightFilter(object):
    """Mask regions based on player position and a 1D (polar) line-of-sight texture
    """
    def __init__(self, texture, pos):
        self.vshader = Function("""
            void line_of_sight() {
                vec4 polar_pos = $transform(vec4($pos, 1));
                if( polar_pos.x > 0.999 ) {
                    polar_pos.x = 0.001;
                }
                vec4 c = texture2D($texture, vec2(polar_pos.x, 0.5));
                float depth = c.r;
                if( polar_pos.y > depth+0.5 ) {
                    $mask = vec3(0.5, 0.5, 1);  // out-of-sight objects turn blue
                }
                else {
                    $mask = vec3(1, 1, 1);
                }
            }
        """)
        self.fshader = Function("""
            void apply_texture_mask() {
                gl_FragColor *= vec4($mask,1);
            }
        """)
        self.center = STTransform()
        self.transform = STTransform(scale=(0.5/np.pi, 1, 0), translate=(0.5, 0, 0)) * PolarTransform().inverse * self.center 
        
        self.vshader['pos'] = pos
        self.vshader['transform'] = self.transform
        self.vshader['texture'] = texture
        self.vshader['mask'] = Varying('mask', dtype='vec3')
        self.fshader['mask'] = self.vshader['mask']

    def set_player_pos(self, pos):
        self.center.translate = (-pos[0], -pos[1])

    def _attach(self, visual):
        self._visual = visual
        vhook = visual._get_hook('vert', 'post')
        vhook.add(self.vshader(), position=3)
        fhook = visual._get_hook('frag', 'post')
        fhook.add(self.fshader(), position=3)


class SightRenderer(object):
    """For computing 1d (polar) line of sight and shadows on GPU
    """
    def __init__(self, scene, opacity, size=(100, 1000)):
        self.scene = scene
        self.size = size
        self.tex = vispy.gloo.Texture2D(shape=size+(4,), format='rgba', interpolation='linear', wrapping='repeat')
        self.fbo = vispy.gloo.FrameBuffer(color=self.tex, depth=vispy.gloo.RenderBuffer(size))
        
        vert = """
            #version 120

            attribute vec3 position;
            uniform float scale;
            uniform float underfill;
            varying vec3 depth;
            
            uniform sampler2D opacity;
            uniform vec2 opacity_size;
            
            void main (void) {
                vec3 cpos = position;
                float alpha = texture2D(opacity, (cpos.xy+vec2(0.5, 0.5)) / opacity_size).r;
                if( alpha > 0 ) {
                    vec4 center = $transform(vec4(cpos, 1));
                    
                    // Determine the min/max azimuthal angle occupied by this wall
                    float min_theta = center.x;
                    float max_theta = center.x;
                    
                    // Check for connection with adjacent walls, extend 
                    for( int i=0; i<2; i++ ) {
                        for( int j=-1; j<2; j+=2 ) {
                            vec3 dx = vec3(0, 0, 0);
                            dx[i] = j;
                            vec3 pos2 = cpos + dx;
                            float alpha2 = texture2D(opacity, (pos2.xy+vec2(0.5, 0.5)) / opacity_size).r;
                            if( alpha2 > 0 ) {
                                dx[i] = j/1.95;  // 1.95 gives a small amount of overlap to prevent gaps
                            }
                            else {
                                dx[i] = j/4.;  // unconnected walls still have some width
                            }
                            vec4 polar_pos = $transform(vec4(cpos + dx, 1));
                            if( polar_pos.x - center.x > 1 ) {
                                // point wraps around between -pi and +pi
                                polar_pos.x -= 2;
                            }
                            else if( center.x - polar_pos.x > 1 ) {
                                polar_pos.x += 2;
                            }
                            min_theta = min(min_theta, polar_pos.x);
                            max_theta = max(max_theta, polar_pos.x);
                        }
                    }
                    if( min_theta < -1 ) {
                        min_theta += 2;
                        max_theta += 2;
                        center.x += 2;
                    }
                    
                    float theta = (min_theta + max_theta) / 2.0;
                    theta = (theta + 1) * underfill - 1;  // compress theta range to avoid overflowing the right edge
                    gl_Position = vec4(theta, 0, center.y/1000., 1);
                    gl_PointSize = scale * underfill * abs(max_theta - min_theta);
                    
                    // encode depth as rgb
                    float r = int(center.y / 256.) / 255.;
                    float g = int(center.y - (r*256)) / 255.;
                    float b = center.y - int(center.y);
                    depth = vec3(r, g, b);
                }
                else {
                    // Not a wall
                    gl_Position = vec4(-2, -2, -2, 1);
                    gl_PointSize = 0;
                }
            }
        """
        
        frag = """
            #version 120
            
            varying vec3 depth;
            
            void main (void) {
                gl_FragColor = vec4(depth, 1);
            }
        """
        
        self.program = ModularProgram(vert, frag)
        self.underfill = 0.9  # Need to underfill the x axis because some points straddle the border between -pi and +pi
        self.program['underfill'] = self.underfill
        self.center = STTransform()
        self.transform = STTransform(scale=(1. / np.pi, 1, 1)) * PolarTransform().inverse * self.center 
        self.program.vert['transform'] = self.transform
        self.program['scale'] = self.size[1] / 2.0
        self.program['wrap'] = self.underfill
        self.program['opacity'] = opacity
        self.program['opacity_size'] = opacity.shape[:2][::-1]
        
    def render(self, pos):
        """Compute distance from pos to nearest object in all directions.
        
        Returns an array of shape (1, N), where [0,0] gives the distance to the
        nearest object at theta=-pi, and [0,-1] gives the nearest object distance
        at theta=+pi.
        
        Strategy is:
        
        1) draw all opaque points mapped into polar coordinates 
           (x=theta, z=depth), with depth encoded as fragment rgb
        2) points that straddle the +pi/-pi boundary are shifted to +pi, and
           the texture is expanded to prevent clipping these overflow fragments
        3) after rendering, the texture is downloaded and decoded, and overflow
           fragments are wrapped back to the left side of the texture
        """
        self.center.translate = (-pos[0], -pos[1])
        self.program['position'] = self.scene.txt.shared_program['position']
        with self.fbo:
            vispy.gloo.clear(color=(0, 0, 0), depth=True)
            vispy.gloo.set_state(depth_test=True)
            vispy.gloo.set_viewport(0, 0, *self.size[::-1])
            self.program.draw(mode='points', check_error=True)
            vispy.gloo.set_viewport(0, 0, *self.scene.canvas.size)
            img = self.fbo.read()
        
        # decode distance from rgb
        dist = img[...,0]*255 + img[...,1] + img[...,2] / 255.        
        
        # wrap from right side overflow back to left side
        i = dist.shape[0] // 2
        j = int(dist.shape[1] * self.underfill)
        v = dist[i, j]
        try:
            j2 = np.argwhere(dist[i, j:] != v)[0,0]
        except IndexError:
            raise Exception("Error: overflowed line-of-sight render buffer :(")
        dist[:, :j2] = dist[:, j:j+j2]
        
        dist = dist[:, :j]
        return dist



class LOSTextureRenderer(object):
    """Converts a 1D polar line-of-sight texture into a 2D (cartesian) shadow map.
    """
    def __init__(self, scene, los_tex, size, supersample=4):
        
        vert = """
            #version 120
            
            attribute vec2 pos;
            varying vec2 v_pos;
            
            void main(void) {
                gl_Position = vec4(pos, 0, 1);
                v_pos = $transform(gl_Position).xy;
            }
        """

        frag = """
            #version 120
            
            varying vec2 v_pos;
            uniform sampler2D los_tex;
            
            void main(void) {
                vec2 polar_pos = $transform(vec4(v_pos, 0, 1)).xy;
                float los_depth = texture2D(los_tex, vec2(polar_pos.x, 0.5)).r;
                float diff = (los_depth+1 - polar_pos.y);
                gl_FragColor = vec4(diff, diff, diff, 1);
            }
        
        """
        self.scene = scene
        self.size = (size[0]*supersample, size[1]*supersample)
        self.vertices = np.array([[-1, -1], [1, -1], [-1, 1], [-1, 1], [1, -1], [1, 1]], dtype='float32')
        self.program = ModularProgram(vert, frag)
        self.program['pos'] = self.vertices
        self.program['los_tex'] = los_tex
        self.program.vert['transform'] = STTransform(scale=(size[1]/2., size[0]/2.)) * STTransform(translate=(1, 1))
        self.center = STTransform()
        self.program.frag['transform'] = STTransform(scale=(0.5 / np.pi, 1, 1), translate=(0.5, 0, 0)) * PolarTransform().inverse * self.center 
        self.tex = vispy.gloo.Texture2D(shape=self.size+(4,), format='rgba', interpolation='linear')
        self.fbo = vispy.gloo.FrameBuffer(color=self.tex, depth=vispy.gloo.RenderBuffer(self.size))
    
    def render(self, pos):
        self.center.translate = (-pos[0]-0.5, -pos[1]-0.5)
        with self.fbo:
            vispy.gloo.clear(color=(0, 0, 0), depth=True)
            vispy.gloo.set_state(depth_test=True)
            vispy.gloo.set_viewport(0, 0, *self.size[::-1])
            self.program.draw(mode='triangles', check_error=True)
            vispy.gloo.set_viewport(0, 0, *self.scene.canvas.size)
            img = self.fbo.read()
        
        return img
    


class ShadowRenderer(object):
    """For computing 2D shadows
    """
    def __init__(self, scene, opacity, supersample=1):
        self.scene = scene
        self.size = (opacity.shape[0] * supersample, opacity.shape[1] * supersample)
        self.fbo = vispy.gloo.FrameBuffer(color=vispy.gloo.RenderBuffer(self.size), 
                                          depth=vispy.gloo.RenderBuffer(self.size))
        
        vert = """
            #version 330 compatibility

            in vec2 ij;
            
            void main (void) {
                gl_Position = vec4(ij, 0, 1);
            }
        """
        
        geom = """
            #version 330 compatibility
        
            layout (points) in;
            layout (triangle_strip, max_vertices=10) out;

            uniform sampler2D opacity;
            uniform vec2 opacity_size;
            uniform vec2 center;
            
            void main (void) {
                vec4 pos = gl_in[0].gl_Position;
                
                // first get opacity of this block and its neighbors
                float opaque[5];
                vec2 dij[5];
                dij[0] = vec2(0, 0);
                dij[1] = vec2(-1, 0);
                dij[2] = vec2(1, 0);
                dij[3] = vec2(0, -1);
                dij[4] = vec2(0, 1);
                
                vec2 uv;
                for( int i=0; i<5; i++ ) {
                    uv = (pos.xy + 0.5 + dij[i]) / opacity_size;
                    opaque[i] = texture(opacity, uv).r;
                }
                    
                // now construct shadows
                float w = 2./3.;
                dij[0] = vec2(w, w);
                dij[1] = vec2(1-w, w);
                dij[2] = vec2(1-w, 1-w);
                dij[3] = vec2(w, 1-w);
                dij[4] = vec2(w, w);
                
                // decide based on opacity of neighbors whether to join walls
                if( opaque[1] > 0.5 ) {
                    dij[0].x = 0;
                    dij[3].x = 0;
                    dij[4].x = 0;
                }
                if( opaque[2] > 0.5 ) {
                    dij[1].x = 1;
                    dij[2].x = 1;
                }
                if( opaque[3] > 0.5 ) {
                    dij[0].y = 0;
                    dij[1].y = 0;
                    dij[4].y = 0;
                }
                if( opaque[4] > 0.5 ) {
                    dij[2].y = 1;
                    dij[3].y = 1;
                }
                
                vec4 pos2;
                if( opaque[0] >= 1 ) {
                    for( int n=0; n<5; n++ ) {
                        pos2 = (pos + vec4(dij[n], 0, 0));
                        vec2 dx = normalize(pos2.xy - (center + 0.5));
                        //pos2 += vec4(dx * .5, 0, 0);
                        gl_Position = pos2 * 2 / vec4(opacity_size, 1, 1) - 1;
                        EmitVertex();
                        pos2 += vec4(dx * 1000, 0, 0);
                        //pos2 = pos + vec4(0.5, 0.5, 0, 0);
                        gl_Position = pos2 * 2 / vec4(opacity_size, 1, 1) - 1;
                        EmitVertex();
                    }
                    EndPrimitive();
                }
                    
            }
        """
        
        frag = """
            #version 330 compatibility
            
            void main (void) {
                gl_FragColor = vec4(0, 0, 0, 1);
            }
        """
        
        self.program = ModularProgram(vert, frag, gcode=geom)
        self.program['opacity'] = vispy.gloo.Texture2D(opacity, format='luminance', interpolation='nearest')
        self.program['opacity_size'] = opacity.shape[:2][::-1]
        
        self.program['ij'] = np.mgrid[0:opacity.shape[1], 0:opacity.shape[0]].astype('float32').transpose(1, 2, 0)
        
        
    def render(self, pos):
        """
        """
        self.program['center'] = pos
        with self.fbo:
            vispy.gloo.clear(color=(1, 1, 1))
            vispy.gloo.set_viewport(0, 0, *self.size[::-1])
            #vispy.gloo.set_state(cull_face=True)
            self.program.draw(mode='points', check_error=True)
            vispy.gloo.set_viewport(0, 0, *self.scene.canvas.size)
            img = self.fbo.read()
        
        return img[::-1]





class Console(object):
    def __init__(self, shape):
        self.shape = shape
        
        self.view = vispy.scene.widgets.ViewBox(border_color=(1, 1, 1, 0.2), bgcolor=(0, 0, 0, 0.4))
        self.view.camera = 'panzoom'
        self.view.camera.rect = vispy.geometry.Rect(-0.5, -0.5, shape[1], shape[0])
        
        # generate a texture for each character we need
        self.atlas = CharAtlas(size=32)
        ascii_chars = [chr(i) for i in range(0x20, 128)]
        self.atlas.add_chars(ascii_chars)
        
        # create sprites visual
        self.txt = Sprites(self.atlas, sprite_size=(1, 1), point_cs='visual', parent=self.view.scene)
        self.txt.update_gl_state(depth_test=False)
        
        self.txt_sprites = self.txt.add_sprites(shape)
        self.txt_sprites.sprite = 0
        
        pos = np.zeros(shape + (3,), dtype='float32')
        pos[...,:2] = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)
        self.txt_sprites.position = pos
        
        fgcolor = np.ones(shape + (4,), dtype='float32')
        fgcolor[...,3] = 0.5
        self.txt_sprites.fgcolor = fgcolor
        
        bgcolor = np.zeros(shape + (4,), dtype='float32')
        bgcolor[..., 3] = 0.5
        self.txt_sprites.bgcolor = bgcolor
        
        self.lines = []

    def write(self, txt):
        self.lines.extend(txt.split('\n'))
        self.update_text()
        
    def update_text(self):
        sprites = np.zeros(self.shape, dtype='uint8')
        sprites[:] = ord(' ')
        for i in range(min(self.shape[0], len(self.lines))):
            line = np.fromstring(self.lines[-i-1].encode('ascii'), dtype='ubyte')
            sprites[i, :len(line)] = line[:self.shape[1]]
        self.txt_sprites.sprite = sprites - 0x20
        
