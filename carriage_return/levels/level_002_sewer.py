"""Level 2, the sewer: six hallways forming a rough grid, from a fixed seed.

The player drops in through a hole in the ceiling (the one-way hole down from
home, wired here) and leaves by the stairs down at the far end. Daylight enters
two ways: a barred window (grate) capping Hall 5, and the ceiling hole itself,
whose shaft is carried by the hole's sewer end like a torch's flame.
"""
import numpy as np

from ..item import Torch
from ..light import ArrayLight, PointLight
from ..maze import Maze
from ..portal import Hole
from ..units import lm, lx
from ..world import Level

#: The sewer is generated from a fixed seed so it is the same every game; the
#: layout (and therefore the rendered image) is a screenshot regression baseline.
SEED = 20240719

#: Hallways are ``2*HALF+1`` cells wide.
HALF = 1

#: The cool daylight tint shared by the grate window and the ceiling hole.
SKY = np.array([0.8, 0.9, 1.0])


def _generate(blocktypes, seed=SEED):
    """Carve the sewer maze from *seed*; return ``(maze, locations, grate_spot)``.

    ``locations`` holds the ``hole`` (west end of Hall 1, the ceiling opening),
    the ``stairs_down`` (far end of Hall 6) and the ``grate`` window cell.
    ``grate_spot`` is the float mask of the cells the barred window's daylight
    strikes, ready for an :class:`~.light.ArrayLight`. Layout, as carved:

    - **Hall 1** -- the long E-W spine; the player drops in at its west end.
    - **Halls 2, 3** -- N-S corridors crossing Hall 1 at random-ish points
      (Hall 2 at 20-50% of its length, Hall 3 at 70-90%), each running a
      random distance both north and south of the spine.
    - **Hall 4** -- a long E-W run at the southmost row either N-S hall reaches,
      so it meets at least the corridor that runs furthest south.
    - **Hall 5** -- a long E-W run at the northmost row that still meets *both*
      N-S halls (as far north as the shorter northern arm reaches). Its east
      end is capped by an impassable grate open to the daylight outside.
    - **Hall 6** -- a N-S corridor dropping south off Hall 4 (and nothing else)
      to a dead end holding the stairs down.

    Lengths come from the seeded generator and are clamped so nothing runs off
    the edge, keeping the layout valid for any seed rather than only for this
    one. Connectivity is by construction: every hall meets the spine, directly
    or through another hall, so the stairs are always reachable from the hole.
    """
    rng = np.random.RandomState(seed)
    shape = (91, 141)         # rows, cols; a generous canvas, cropped at the end
    rows, cols = shape
    half = HALF
    margin = half + 1         # never carve into the outer wall

    maze = Maze.filled(shape, blocktypes, 'wall', obj_name='sewer')
    blocks = maze.blocks
    path = blocktypes.id_of('path')
    grate = blocktypes.id_of('grate')

    def carve_h(y, xa, xb):
        lo, hi = min(xa, xb), max(xa, xb)
        blocks[y - half:y + half + 1, lo:hi + 1] = path

    def carve_v(x, ya, yb):
        lo, hi = min(ya, yb), max(ya, yb)
        blocks[lo:hi + 1, x - half:x + half + 1] = path

    # Hall 1: the E-W spine. The hole drops in at the west end.
    y1 = rows // 2
    x_w = margin + 2
    len1 = min(int(rng.randint(60, 90)), cols - margin - 1 - x_w)
    x_e = x_w + len1
    carve_h(y1, x_w, x_e)
    hole_pos = (x_w + 3, y1)

    # Halls 2, 3: N-S, crossing Hall 1 at 20-50% and 70-90% of its length, each
    # running a random distance both north and south of the spine.
    x2 = x_w + int(len1 * rng.uniform(0.20, 0.50))
    x3 = x_w + int(len1 * rng.uniform(0.70, 0.90))
    n2 = min(int(rng.randint(10, 22)), y1 - margin)
    s2 = min(int(rng.randint(10, 22)), rows - margin - 1 - y1)
    n3 = min(int(rng.randint(10, 22)), y1 - margin)
    s3 = min(int(rng.randint(10, 22)), rows - margin - 1 - y1)
    north2, south2 = y1 - n2, y1 + s2
    north3, south3 = y1 - n3, y1 + s3
    carve_v(x2, north2, south2)
    carve_v(x3, north3, south3)

    # Hall 4: long E-W, at the southmost point either N-S hall reaches, so it
    # meets at least the corridor running furthest south.
    y4 = max(south2, south3)
    x4_w = max(margin + 1, min(x2, x3) - int(rng.randint(6, 14)))
    x4_e = min(cols - margin - 1, max(x2, x3) + int(rng.randint(6, 14)))
    carve_h(y4, x4_w, x4_e)

    # Hall 5: long E-W, at the northmost row that still meets both N-S halls --
    # as far north as the shorter of the two northern arms reaches -- running
    # east past Hall 3 to a barred window on the outside.
    y5 = max(north2, north3)
    x5_w = min(x2, x3)
    x5_e = min(cols - margin - 2, max(x2, x3) + int(rng.randint(12, 20)))
    carve_h(y5, x5_w, x5_e)

    # The grate capping Hall 5: '#' bars a little taller than the hallway, so it
    # reads as a window. Impassable, but see-through (opacity 0), and cropped in
    # below with the floor so its bright bars survive the crop.
    grate_x = x5_e + 1
    blocks[y5 - half - 1:y5 + half + 2, grate_x] = grate

    # Hall 6: N-S, dropping south off Hall 4 (and crossing nothing else, since
    # everything else lies at or north of Hall 4) to a dead end with the stairs.
    x6 = (x2 + x3) // 2
    if x6 in (x2, x3):
        x6 += 2
    len6 = min(int(rng.randint(8, 16)), rows - margin - 1 - y4)
    y6 = y4 + len6
    carve_v(x6, y4, y6)
    stairs_pos = (x6, y6)

    # Crop to the carved area (floor *and* grate) plus a one-cell wall margin.
    # The generator works on a canvas big enough for any seed, but the level
    # that survives should not be mostly solid rock: the sight field is
    # supersampled per maze cell, so unreachable rock costs real memory and time
    # every frame. The grate is included in the mask so its bright bars are not
    # cropped away with the surrounding wall.
    mask = (blocks == path) | (blocks == grate)
    ys, xs = np.nonzero(mask)
    y_lo, y_hi = ys.min() - 1, ys.max() + 2
    x_lo, x_hi = xs.min() - 1, xs.max() + 2
    cropped = Maze(blocks[y_lo:y_hi, x_lo:x_hi].copy(), blocktypes, obj_name='sewer')
    shift = lambda p: (p[0] - x_lo, p[1] - y_lo)

    # The cells the barred window's daylight strikes, in cropped coordinates:
    # the floor just inside the bars, and the bars themselves.
    gx, gy = shift((grate_x, y5))
    xb = shift((x5_e - 2, 0))[0]
    grate_spot = np.zeros(cropped.shape, dtype='float32')
    grate_spot[gy - half:gy + half + 1, xb:gx + 1] = 1.0   # floor inside the bars
    grate_spot[gy - half - 1:gy + half + 2, gx] = 1.0      # the barred window

    locations = {
        'hole': shift(hole_pos),
        'stairs_down': shift(stairs_pos),
        'grate': (gx, gy),
    }
    return cropped, locations, grate_spot


def build_level(scene):
    """Build the sewer, its daylight, and the one-way hole down from home."""
    world = scene.world
    maze, locations, grate_spot = _generate(world.blocktypes)

    level = Level('sewer', maze)
    level.locations.update(locations)

    # Daylight through the grate: the same cool sky as the ceiling hole but a
    # far wider aperture, pouring in and streaming down Hall 5. Two lights
    # mirror the hole's model below -- an ArrayLight for the bright patch the
    # sky strikes (the bars and the floor just inside them), and a PointLight,
    # much brighter than the hole's, for the wash down the hall. These belong to
    # the map, not to anything that moves, so they are pinned to the maze like
    # home's daylight rather than carried by a portal.
    gx, gy = locations['grate']
    maze.add_light(ArrayLight(maze, grate_spot, color=SKY * 100 * lx), pos=(gx, gy))
    maze.add_light(PointLight(maze, color=SKY * 300 * lm, brightness=1.0), pos=(gx, gy))

    # Down the hole from home and you cannot climb back. The home end is an
    # ordinary hole in the floor -- an 'O' you drop through. The sewer end is
    # the ceiling opening you land under: not enterable (step onto it and it
    # still says the way up is out of reach), and invisible (char=None) because
    # there is no hole in the sewer floor -- just the patch of floor the
    # daylight lands on.
    home = world.levels['home']
    top = Hole(location=(home.maze, home.locations['hole']), scene=scene)
    bottom = Hole(location=(maze, locations['hole']), scene=scene,
                  enterable=False, char=None)

    # The daylight is the sewer hole's own light, carried by the end like a
    # torch's flame. Physically it is a thin shaft of the *same* sun that lights
    # home, coming through a small hole in the ceiling: the one cell the beam
    # strikes is outdoor-bright, while the fraction that scatters off that floor
    # patch and bounces down the corridor is only ~1/100 of it. That 100:1 is
    # the whole point of the absolute scale -- home daylight is ~100x the light
    # that reaches you down the hole. Two lights express it, in lux/lumens:
    #
    #  - an ArrayLight for the beam spot: value 1.0 on the struck cell, so its
    #    colour *is* the illuminance there in lux. A cool daylight blue at
    #    ~50000 lux -- the same open-sun level home is lit to -- so the struck
    #    cell reads as a patch of full daylight.
    #  - a PointLight for the scattered wash: colour is luminous flux in lumens
    #    (see PointLight), falling off as E = Phi/(4*pi*r^2) lux. ~27000 lm on a
    #    cool tint lands ~500 lux (home/100) a couple of cells out.
    #
    # Magnitudes are starting points, tunable in the visual pass; only the 100:1
    # home:hole ratio is load-bearing.
    hx, hy = locations['hole']
    spot = np.zeros(maze.shape, dtype='float32')
    spot[hy, hx] = 1.0
    bottom.lights = [
        ArrayLight(bottom, spot, color=SKY * 100 * lx),
        PointLight(bottom, color=SKY * 50 * lm, brightness=1.0),
    ]
    world.link(top, bottom)

    # The stairs down get a torch so they are findable from a distance; the
    # ceiling opening the player fell through is lit by its own daylight.
    sx, sy = locations['stairs_down']
    Torch(location=(maze, (sx - 1, sy)), scene=scene)

    return level
