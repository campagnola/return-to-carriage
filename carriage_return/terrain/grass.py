"""Patchy grass colour: sun-parched patches mixed lightly into a grass floor."""
import numpy as np

from ..random.spectral import gaussian_blur_sigma_freq, gaussian_spectrum, make_noise


#: Deep green fading through yellow to brown, for patches of parched grass.
GRASS_WASH_RAMP_T = (0.0, 0.5, 1.0)
GRASS_WASH_RAMP_RGB = (
    (0.02, 0.18, 0.03),   # deep green
    (0.45, 0.40, 0.06),   # yellow
    (0.28, 0.17, 0.06),   # brown
)
GRASS_WASH_AMOUNT = 0.15

#: Spatial scale (px) of the broad sun-parched patches -- the low-frequency
#: component that carries the ramp above.
GRASS_WASH_LF_SCALE = 8.0

#: Spatial scale (px) of the fine speckle blended into the LF patches --
#: high-frequency detail so the wash reads as individual blades/tufts rather
#: than a smooth colour gradient.
GRASS_WASH_HF_SCALE = 1.2

#: How much the HF speckle perturbs the LF field before ramp lookup, relative
#: to the LF field's own [0, 1] range -- kept small so it adds texture without
#: breaking up the broad patches themselves.
GRASS_WASH_HF_WEIGHT = 0.12


def grass_wash(rng, shape):
    """A patchy plant-matter colour field: deep green fading through yellow to
    brown, following broad soft blobs of noise -- sun-parched patches in the
    grass, in no particular arrangement -- with a fine high-frequency speckle
    layered on top for texture. Meant to be mixed lightly into the grass
    blocktype's flat colour via :meth:`~..maze.Maze.wash_bg_color` (see
    :func:`paint_grass_wash`).
    """
    lf = make_noise(shape, rng, gaussian_spectrum(gaussian_blur_sigma_freq(GRASS_WASH_LF_SCALE)), stdev=1.0)
    hf = make_noise(shape, rng, gaussian_spectrum(gaussian_blur_sigma_freq(GRASS_WASH_HF_SCALE)), stdev=1.0)
    lf -= lf.min()
    lf /= lf.max()
    field = np.clip(lf + GRASS_WASH_HF_WEIGHT * hf, 0.0, 1.0)
    channels = [np.interp(field, GRASS_WASH_RAMP_T, [rgb[i] for rgb in GRASS_WASH_RAMP_RGB])
                for i in range(3)]
    return np.stack(channels, axis=-1).astype('float32')


def paint_grass_wash(maze, bt, rng):
    """Mix a patchy grass wash into every grass cell still showing on *maze*.

    Restricted to cells currently painted as grass, so it follows whatever
    footprint is left once paths, rivers and buildings have been laid down --
    call this last.
    """
    mask = maze.blocks == bt.id_of('grass')
    wash = grass_wash(rng, maze.shape)
    maze.wash_bg_color(wash, GRASS_WASH_AMOUNT, mask=mask)
