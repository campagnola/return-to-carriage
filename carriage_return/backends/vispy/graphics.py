import numpy as np
from PyQt5 import QtGui
import vispy.visuals, vispy.scene, vispy.gloo
from vispy.visuals.shaders import ModularProgram, Function
from vispy.visuals.transforms import STTransform, NullTransform

from ...tone_mapping import GLSL_REINHARD_TONEMAP, MEMORY_TINT


# load support for opengl 3 features
vispy.gloo.gl.use_gl('gl+')

#: Bootstrap exposure for the shader before the player's eye adaptation is ever
#: pushed. The eye establishes its real reference from the first scene it sees
#: (see :mod:`...adaptation`), and the renderer overrides this uniform every
#: composite from ``player.adaptation.exposure``; this only has to keep a frame
#: drawn before that first update from being black or blown out. Order of
#: key / (daylight-adapted reflected luminance ~ tens of cd/m^2).
_DEFAULT_EXPOSURE = 0.002


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
        attribute vec3 fg_emission;
        attribute vec3 bg_emission;

        varying float v_sprite;
        varying vec4 v_fgcolor;
        varying vec4 v_bgcolor;
        varying vec3 v_fg_emission;
        varying vec3 v_bg_emission;
        varying vec2 v_point_coord_scale;
        varying vec2 v_point_size;

        void main (void) {
            v_fgcolor = fgcolor;
            v_bgcolor = bgcolor;
            v_fg_emission = fg_emission;
            v_bg_emission = bg_emission;
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
        varying vec3 v_fg_emission;
        varying vec3 v_bg_emission;
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
            //EMISSION_WRITE
        }
    """

    vertex_shader_3 = """
        #version 330 compatibility

        in vec3 position;
        in float sprite;
        in vec4 fgcolor;
        in vec4 bgcolor;
        in vec3 fg_emission;
        in vec3 bg_emission;

        out float v_sprite;
        out vec4 v_fgcolor;
        out vec4 v_bgcolor;
        out vec3 v_fg_emission;
        out vec3 v_bg_emission;
        out vec2 v_sprite_size;

        void main (void) {
            v_fgcolor = fgcolor;
            v_bgcolor = bgcolor;
            v_fg_emission = fg_emission;
            v_bg_emission = bg_emission;
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
        in vec3 v_fg_emission[];
        in vec3 v_bg_emission[];
        in vec2 v_sprite_size[];

        out float f_sprite;
        out vec4 f_fgcolor;
        out vec4 f_bgcolor;
        out vec3 f_fg_emission;
        out vec3 f_bg_emission;
        out vec2 point_coord;
        out vec2 point_size;

        uniform vec2 scale;

        void main(void) {
            f_sprite = v_sprite[0];
            f_fgcolor = v_fgcolor[0];
            f_bgcolor = v_bgcolor[0];
            f_fg_emission = v_fg_emission[0];
            f_bg_emission = v_bg_emission[0];
        
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
        in vec3 f_fg_emission;
        in vec3 f_bg_emission;
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
            const int ss = 3;
            float alpha = 0;
            for (int i=0; i<ss; i++) {
                for (int j=0; j<ss; j++) {
                    vec2 dx = vec2(i, j) / (point_size*ss);
                    tex_coords = vec2(0.00001, 0.00001) + atlas_coords.yx + (pt + dx) * atlas_coords.wz;
                    vec4 tex = texture2D(atlas, tex_coords);
                    alpha += tex.g / (ss*ss);
                }
            }
            
            gl_FragColor = f_fgcolor * alpha + f_bgcolor * (1-alpha);
            //EMISSION_WRITE
        }
    """

    #: Whether this visual's fragment shader stores per-fragment emission for a
    #: tone-mapping filter to read. Off by default (HUD grids draw their glyphs
    #: raw, with no filter attached); the world layer visual turns it on -- see
    #: LayerSpritesVisual -- so an emitter such as a torch flame glows. The
    #: shared global the write targets is declared by TextureMaskFilter, which
    #: is why the write is emitted only when that filter is in play.
    emits = False

    def __init__(self, atlas, sprite_size=(16, 16), point_cs='pixel', method=None):
        if method is None:
            if 'GL_GEOMETRY_SHADER' in vispy.gloo.gl.__dict__:
                method = 'geometry'
            else:
                method = 'point_sprite'

        self.method = method
        self.point_cs = point_cs

        self.sprite_data_items = []

        self.atlas = atlas
        self.atlas.atlas_changed.connect(self._atlas_changed)
        self.position = np.empty((0, 3), dtype='float32')
        self.sprite = np.empty((0,), dtype='uint32')
        self.sprite_size = sprite_size
        self.fgcolor = np.empty((0, 4), dtype='float32')
        self.bgcolor = np.empty((0, 4), dtype='float32')
        # Linear emitted radiance per sprite (RGB, no alpha); 0 for a plain
        # reflective glyph, non-zero for an emitter such as a torch flame.
        self.fg_emission = np.empty((0, 3), dtype='float32')
        self.bg_emission = np.empty((0, 3), dtype='float32')

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
        shaders = tuple(self._resolve_emission(src) for src in shaders)
        vispy.visuals.Visual.__init__(self, *shaders)

        self._draw_mode = 'points'
        self.shared_program['position'] = vispy.gloo.VertexBuffer()
        self.shared_program['sprite'] = vispy.gloo.VertexBuffer()
        self.shared_program['fgcolor'] = vispy.gloo.VertexBuffer()
        self.shared_program['bgcolor'] = vispy.gloo.VertexBuffer()
        self.shared_program['fg_emission'] = vispy.gloo.VertexBuffer()
        self.shared_program['bg_emission'] = vispy.gloo.VertexBuffer()
        
        # blending must be declared here: vispy applies gl_state incrementally
        # per visual, so relying on another visual having enabled blend leaves
        # translucent cell backgrounds opaque depending on draw order
        self.update_gl_state(depth_test=True, blend=True,
                             blend_func=('src_alpha', 'one_minus_src_alpha'))

    def _resolve_emission(self, src):
        """Fill in the fragment shader's ``//EMISSION_WRITE`` marker.

        A visual that emits (``emits`` True) turns the marker into a statement
        that blends fg and bg emission by glyph coverage and stashes the result
        in ``sprite_emission`` for the tone-mapping filter to read; the filter
        is what declares that global, so the write is emitted only when a filter
        is in play. A visual that does not emit (the HUD grids) leaves the
        marker as the comment it already is, so it draws its glyphs with no
        emission and no filter.
        """
        marker = '//EMISSION_WRITE'
        i = src.find(marker)
        if i < 0 or not self.emits:
            return src
        if '#version 120' in src:
            fg_varying = 'v_fg_emission'
            bg_varying = 'v_bg_emission'
        else:
            fg_varying = 'f_fg_emission'
            bg_varying = 'f_bg_emission'
        replacement = 'sprite_emission = %s * alpha + %s * (1.0 - alpha);' % (fg_varying, bg_varying)
        return src.replace(marker, replacement)

    def add_sprites(self, shape):
        """Expand to allow more sprites, return a SpriteData instance with the specified shape.
        """
        if not isinstance(shape, tuple):
            raise TypeError("shape must be a tuple (got %r)" % shape)
        n = np.prod(shape)
        old_size = self._resize(self.position.shape[0] + n)
        sd = SpriteData(self, start=old_size, shape=shape)
        self.sprite_data_items.append(sd)
        return sd

    def _resize(self, n):
        """Resize sprite array, return old size.
        """
        n1 = self.position.shape[0]        
        self.position = np.resize(self.position, (n, 3))
        self.sprite = np.resize(self.sprite, (n,))
        self.fgcolor = np.resize(self.fgcolor, (n, 4))
        self.bgcolor = np.resize(self.bgcolor, (n, 4))
        self.fg_emission = np.resize(self.fg_emission, (n, 3))
        self.bg_emission = np.resize(self.bg_emission, (n, 3))
        self._upload_data()
        return n1

    def data_changed_shape(self):
        size = sum([len(sd) for sd in self.sprite_data_items])
        self._resize(size)
        start = 0
        for sd in self.sprite_data_items:
            sd.set_start(start)
            start += len(sd)
    
    def _upload_data(self):
        self.shared_program['position'].set_data(self.position)
        self.shared_program['sprite'].set_data(self.sprite.astype('float32'))
        self.shared_program['fgcolor'].set_data(self.fgcolor)
        self.shared_program['bgcolor'].set_data(self.bgcolor)
        self.shared_program['fg_emission'].set_data(self.fg_emission)
        self.shared_program['bg_emission'].set_data(self.bg_emission)
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
        self.set_shape(shape, inform_parent=False)
        self.set_start(start)

    def __len__(self):
        return np.prod(self.shape)

    @property
    def position(self):
        start, stop = self.indices
        return self.sprites.position[start:stop].reshape(self.shape + (3,))
    
    @position.setter
    def position(self, p):
        self._position = p
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
        self._sprite = p
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
        self._fgcolor = p
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
        self._bgcolor = p
        start, stop = self.indices
        self.bgcolor[:] = p
        self.sprites.shared_program['bgcolor'][start:stop] = self.bgcolor.view(dtype=[('bgcolor', 'float32', 4)]).reshape(stop-start)
        self.sprites.update()

    @property
    def fg_emission(self):
        start, stop = self.indices
        return self.sprites.fg_emission[start:stop].reshape(self.shape + (3,))

    @fg_emission.setter
    def fg_emission(self, p):
        self._fg_emission = p
        start, stop = self.indices
        self.fg_emission[:] = p
        self.sprites.shared_program['fg_emission'][start:stop] = self.fg_emission.view(dtype=[('fg_emission', 'float32', 3)]).reshape(stop-start)
        self.sprites.update()

    @property
    def bg_emission(self):
        start, stop = self.indices
        return self.sprites.bg_emission[start:stop].reshape(self.shape + (3,))

    @bg_emission.setter
    def bg_emission(self, p):
        self._bg_emission = p
        start, stop = self.indices
        self.bg_emission[:] = p
        self.sprites.shared_program['bg_emission'][start:stop] = self.bg_emission.view(dtype=[('bg_emission', 'float32', 3)]).reshape(stop-start)
        self.sprites.update()

    def set_start(self, start):
        self.indices = (start, start + len(self))
        if self._position is not None:
            self.position = self._position
            self.sprite = self._sprite
            self.fgcolor = self._fgcolor
            self.bgcolor = self._bgcolor
            # optional: only emitting regions ever set these; others keep the 0
            # the visual's arrays default to after a resize
            if self._fg_emission is not None:
                self.fg_emission = self._fg_emission
            if self._bg_emission is not None:
                self.bg_emission = self._bg_emission

    def set_shape(self, shape, inform_parent=True):
        self.shape = shape
        self._position = None
        self._sprite = None
        self._fgcolor = None
        self._bgcolor = None
        self._fg_emission = None
        self._bg_emission = None
        if inform_parent:
            self.sprites.data_changed_shape()


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

    def __getitem__(self, char):
        if char not in self.chars:
            return self.add_chars(char)
        else:
            return self.chars[char]

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
    """Tone-maps the sprites against the level's light and memory fields.

    Tone mapping lives here, per fragment, after the sprite's albedo color has
    been composited: gl_FragColor.rgb is the material color fg*alpha+bg*(1-alpha)
    (treated as linear reflectance) and gl_FragColor.a is coverage. Two textures
    feed it: the light texture supplies raw linear HDR illuminance in rgb and
    the line-of-sight scalar (0..1) in a; the memory texture supplies the
    display-space memory overlay in r. The fragment's outgoing luminance is
    reflected plus emitted -- ``albedo*illuminance + emission`` -- with BOTH
    terms gated by line of sight, then exposed and run through a Reinhard curve
    + display gamma together. An emitter (a torch flame) therefore adds its own
    light on top of what it reflects, and bright light physically blows a
    surface toward white rather than capping at its albedo.

    Gating on line of sight rather than on local light is the point of the two
    textures: emission rides on the sprite as ``sprite_emission``
    (coverage-scaled linear radiance the sprite shader leaves in a shared
    global), and a self-emitting glyph in a cell that is in view but unlit still
    shows, because its emission is gated by line of sight (a) and not by whether
    any light happens to fall on the cell. Outside the player's view line of
    sight is zero, so a glyph behind a wall neither reflects nor glows -- only
    its remembered overlay shows. Exposure is driven by the player's eye
    adaptation (set_exposure, once per frame).
    """
    def __init__(self, light_texture, memory_texture, transform, scale):
        self.fshader = Function("""
            // Coverage-scaled linear emitted radiance, written by the sprite
            // fragment shader's main() and read here. Declared in this filter
            // (a dependency, emitted ahead of main) so it is in scope by the
            // time main writes it; a sprite drawn without this filter attached
            // (the HUD grids) simply never has its value read.
            vec3 sprite_emission;

            void apply_texture_mask() {
                vec4 tex_pos = $transform(gl_FragCoord);
                tex_pos /= tex_pos.w;
                vec4 lgt = texture2D($light, tex_pos.xy);     // rgb = HDR illuminance (ungated), a = line of sight
                float los = lgt.a;
                float mem = texture2D($memory, tex_pos.xy).r; // display-space remembered overlay (masked to unseen)
                vec3 albedo = gl_FragColor.rgb;
                // Outgoing luminance = reflected + emitted; both gated by line
                // of sight (neither is visible through a wall), then exposed and
                // tone-mapped together.
                vec3 refl = albedo * lgt.rgb * los;           // reflected luminance (linear)
                vec3 out_lum = (refl + sprite_emission * los) * $exposure;   // exposed outgoing luminance
                vec3 lit = $reinhard_tonemap(out_lum);         // Reinhard tone curve + display gamma
                gl_FragColor = vec4(lit + mem * $memory_tint, gl_FragColor.a);
            }
        """)
        self.fshader['light'] = light_texture
        self.fshader['memory'] = memory_texture
        # Reinhard curve + display gamma, defined once in tone_mapping.py and
        # shared with the CPU-side memory overlay (Level.update_sight) -- see
        # tone_mapping.reinhard_tonemap, its Python twin.
        self.fshader['reinhard_tonemap'] = Function(GLSL_REINHARD_TONEMAP)
        # sane default exposure so the first frame (before any update pushes the
        # player's adaptation) is valid; see _DEFAULT_EXPOSURE.
        self.fshader['exposure'] = _DEFAULT_EXPOSURE
        self.fshader['memory_tint'] = MEMORY_TINT
        self.scale_tr = STTransform(scale=scale) * STTransform(translate=(0.5, 0.5))
        self.fshader['transform'] = self.scale_tr * transform
        
        self._visual = None
        # the expression added to the visual's hook, kept so that _detach can
        # remove the same object again -- calling self.fshader() a second time
        # would build a different expression and remove nothing
        self._fshader_expr = None

    def set_exposure(self, value):
        """Set the Reinhard exposure scalar (key / adaptation_luminance)."""
        self.fshader['exposure'] = float(value)

    def _attach(self, visual):
        self._visual = visual
        self._fshader_expr = self.fshader()
        fhook = visual._get_hook('frag', 'post')
        fhook.add(self._fshader_expr, position=3)

    def _detach(self, visual):
        """Remove this filter from *visual* (see Visual.detach).

        Needed because the sight mask is sized to the maze, so switching to a
        differently-shaped level detaches the old filter and attaches a new one.
        """
        if self._fshader_expr is not None:
            visual._get_hook('frag', 'post').remove(self._fshader_expr)
            self._fshader_expr = None
        self._visual = None


class ShadowRenderer(object):
    """For computing 2D shadows
    """
    def __init__(self, maze, canvas, supersample=1):
        self.maze = maze
        self.canvas = canvas
        self.size = (maze.shape[0] * supersample, maze.shape[1] * supersample)
        
        # for render to texture
        self.texture = vispy.gloo.Texture2D(shape=self.size+(4,), format='rgba', interpolation='linear', wrapping='repeat')
        self.fbo = vispy.gloo.FrameBuffer(color=self.texture, 
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
        self.program['opacity'] = vispy.gloo.Texture2D(maze.opacity, format='luminance', interpolation='nearest')
        self.program['opacity_size'] = maze.shape[:2][::-1]
        
        corner_coords = np.mgrid[0:maze.shape[1], 0:maze.shape[0]].astype('float32').transpose(1, 2, 0)
        self.program['ij'] = corner_coords.copy()  # copy to prevent warning about discontiguous data
        
    def render(self, pos, read=False):
        """
        """
        self.program['center'] = pos
        img = None
        with self.fbo:
            vispy.gloo.clear(color=(1, 1, 1))
            vispy.gloo.set_viewport(0, 0, *self.size[::-1])
            #vispy.gloo.set_state(cull_face=True)
            self.program.draw(mode='points', check_error=True)

            vispy.gloo.set_viewport(0, 0, *self.canvas.size)
            if read:
                img = self.fbo.read()[::-1]
        
        return img


