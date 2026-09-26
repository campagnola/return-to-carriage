"""``display_value`` and its GLSL twin must agree."""
import numpy as np
import pytest

from carriage_return.tone_mapping import (display_value, pixel_color,
                                          reflected_luminance)


def test_pixel_color_is_display_value_of_reflected_luminance():
    rng = np.random.default_rng(0)
    albedo = rng.uniform(0, 1, (50, 3))
    illuminance = rng.uniform(0, 5000, (50, 3))
    emission = rng.uniform(0, 100, (50, 3))
    for exposure_value in (1e-4, 0.002, 0.04):
        expected = display_value(reflected_luminance(albedo, illuminance),
                                 emission, exposure_value)
        np.testing.assert_array_equal(
            pixel_color(albedo, emission, illuminance, exposure_value), expected)


_VERT = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# One fragment per input sample: reflected in the first texture's rgb,
# emission in the second's rgb and exposure in its alpha.
_FRAG = """
uniform sampler2D u_reflected;
uniform sampler2D u_emission_exposure;
uniform vec2 u_size;
void main() {
    vec2 uv = gl_FragCoord.xy / u_size;
    vec3 reflected = texture2D(u_reflected, uv).rgb;
    vec4 ee = texture2D(u_emission_exposure, uv);
    gl_FragColor = vec4($display_value(reflected, ee.rgb, ee.a), 1.0);
}
"""


def _gl_canvas():
    """A hidden vispy canvas, or skip if no GL context can be made here."""
    app = pytest.importorskip('vispy.app')
    try:
        canvas = app.Canvas(show=False, size=(8, 8))
        canvas.set_current()
    except Exception as exc:  # no display / GL driver in this environment
        pytest.skip('cannot create an OpenGL context: %s' % exc)
    return canvas


def test_glsl_display_value_matches_python_twin():
    from vispy import gloo
    from vispy.visuals.shaders import ModularProgram
    from carriage_return.backends.vispy.graphics import display_value_function

    # Grid of inputs covering black, the Reinhard knee and deep saturation,
    # at exposures spanning the eye's adaptation range.
    reflected = np.array([0.0, 1e-3, 0.5, 3.0, 40.0, 2000.0])
    emission = np.array([0.0, 0.25, 10.0, 500.0])
    exposures = np.array([1e-4, 0.002, 0.04, 1.0])
    r, e, x = (a.ravel() for a in np.meshgrid(reflected, emission, exposures,
                                              indexing='ij'))
    # Give the three channels distinct values, so a channel mix-up shows.
    refl_rgb = np.stack([r, r * 0.5, r * 0.25], axis=-1)
    emis_rgb = np.stack([e * 0.25, e, e * 0.5], axis=-1)
    n = r.size
    size = (n, 1)  # (width, height)

    canvas = _gl_canvas()
    try:
        refl_tex = gloo.Texture2D(refl_rgb.reshape(1, n, 3).astype('float32'),
                                  format='rgb', internalformat='rgb32f',
                                  interpolation='nearest')
        ee = np.concatenate([emis_rgb, x[:, None]], axis=-1)
        ee_tex = gloo.Texture2D(ee.reshape(1, n, 4).astype('float32'),
                                format='rgba', internalformat='rgba32f',
                                interpolation='nearest')
        out = gloo.Texture2D(shape=(1, n, 4), format='rgba',
                             internalformat='rgba32f')
        fbo = gloo.FrameBuffer(color=out)

        program = ModularProgram(_VERT, _FRAG)
        program.frag['display_value'] = display_value_function()
        program['a_position'] = np.array(
            [[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype='float32')
        program['u_reflected'] = refl_tex
        program['u_emission_exposure'] = ee_tex
        program['u_size'] = size

        with fbo:
            gloo.set_viewport(0, 0, *size)
            gloo.set_state(blend=False, depth_test=False)
            gloo.clear(color=(0, 0, 0, 0))
            program.draw('triangle_strip')
            got = gloo.read_pixels((0, 0) + size, out_type=np.float32)
    finally:
        canvas.close()

    got = got.reshape(n, 4)[:, :3]
    expected = display_value(refl_rgb, emis_rgb, x[:, None])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
