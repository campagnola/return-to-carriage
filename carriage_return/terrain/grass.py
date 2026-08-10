"""Patchy grass colour: sun-parched patches mixed lightly into a grass floor."""
import numpy as np
from scipy.ndimage import gaussian_filter


#: Deep green fading through yellow to brown, for patches of parched grass.
GRASS_WASH_RAMP_T = (0.0, 0.5, 1.0)
GRASS_WASH_RAMP_RGB = (
    (0.02, 0.18, 0.03),   # deep green
    (0.45, 0.40, 0.06),   # yellow
    (0.28, 0.17, 0.06),   # brown
)
GRASS_WASH_AMOUNT = 0.15


def grass_wash(rng, shape):
    """A patchy plant-matter colour field: deep green fading through yellow to
    brown, following broad soft blobs of noise -- sun-parched patches in the
    grass, in no particular arrangement. Meant to be mixed lightly into the
    grass blocktype's flat colour via :meth:`~..maze.Maze.wash_bg_color`
    (see :func:`paint_grass_wash`).
    """
    noise = rng.uniform(0.0, 1.0, size=shape).astype('float32')
    blurred = gaussian_filter(noise, sigma=8.0)
    blurred -= blurred.min()
    blurred /= blurred.max()
    channels = [np.interp(blurred, GRASS_WASH_RAMP_T, [rgb[i] for rgb in GRASS_WASH_RAMP_RGB])
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
