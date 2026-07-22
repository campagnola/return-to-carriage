"""Builders for the levels the game ships with, and the world that joins them.

Each per-level builder takes the world's shared :class:`~.blocktypes.BlockTypes`
table and returns a :class:`~.world.Level` plus whatever positions the caller
has to place things at (portal mouths, torch stands). Positions are ``(x, y)``,
matching entity location slots.

:func:`build_world` assembles the three levels, the portals between them,

    home --(hole, one way)--> sewer --(stairs)--> dungeon

and everything that belongs to the map: the scroll, the torches, the shaft of
daylight in the sewer, and the monster. All that content lives here so it is
not mirror-edited across the game and the screenshot harness. Only the player
and its inventory are placed by the caller (see :func:`.game.new_game`).

Game-side module: no rendering library may be imported here.
"""
import numpy as np

from .light import AmbientLight, ArrayLight, PointLight
from .maze import Maze
from .monster import Monster
from .item import Scroll, Torch
from .portal import Hole, StairsDown, StairsUp
from .units import klx, klm, lm, lx
from .world import Level, World


#: Where the player begins, on the home level.
HOME_START = (3, 5)

#: Where the sewer stairs land in the dungeon -- the game's historical start.
DUNGEON_ARRIVAL = (7, 7)

#: Torch positions (dungeon ``(row, col)`` cells). Torch count and placement
#: change the lighting and therefore the rendered image, so this list is also
#: the screenshot regression baseline.
TORCH_POSITIONS = [
    (17, 8),
    (3, 8),
    (9, 30),
    (32, 41),
    (32, 45),
    (43, 39),

    (15, 75),
    (28, 75),
    (40, 75),
    (52, 75),
    (15, 82),
    (28, 82),
    (40, 82),
    (52, 82),
]


def build_home(blocktypes):
    """A small walled room with a hole in the floor. Returns (level, hole_pos).

    This is where the player starts and the only way out is down. The room's
    sunlight belongs to the map, and a map light only registers with a level
    that already exists (see :meth:`Light._register`), so the level is built
    here, before the light is added, rather than left to :func:`build_world`.
    """
    maze = Maze.filled((40, 120), blocktypes, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = blocktypes.id_of('path')

    hole_pos = (11, 5)
    level = Level('home', maze)

    # The even wash of daylight through the roof. A map light: it belongs to
    # the room, not to anything that moves, and needs no scene -- it announces
    # any change through its own signal, which the level handles.
    maze.add_light(AmbientLight(maze, color=(50000,) * 3), pos=(0, 0))

    return level, hole_pos


def build_sewer(blocktypes, seed=20240719):
    """Six hallways forming a rough grid, generated from a fixed *seed*.

    Returns ``(level, hole_pos, stairs_pos)``. The hole is at the west end of
    Hall 1 -- the ceiling opening the player falls in through -- and the stairs
    down are at the far end of Hall 6. The layout, in the order it is carved:

    - **Hall 1** -- the long E-W spine; the player drops in at its west end.
    - **Halls 2, 3** -- N-S corridors crossing Hall 1 at random-ish points
      (Hall 2 at 20-50% of its length, Hall 3 at 70-90%), each running a
      random distance both north and south of the spine.
    - **Hall 4** -- a long E-W run at the southmost row either N-S hall reaches,
      so it meets at least the corridor that runs furthest south.
    - **Hall 5** -- a long E-W run at the northmost row that still meets *both*
      N-S halls (as far north as the shorter northern arm reaches). Its east
      end is capped by an impassable grate open to the daylight outside, which
      streams down the hall (lights added below).
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
    half = 1                  # hallways are 2*half+1 cells wide
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
    hole_pos = (x_w, y1)

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
    # reads as a window. Impassable, but see-through (opacity 0), and cropped
    # in below with the floor so its bright bars survive the crop.
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
    level = Level('sewer', cropped)

    # Daylight through the grate: the same cool sky as the hole shaft (see
    # build_world) but a far wider aperture, pouring in and streaming down Hall
    # 5. Two lights mirror the hole's model -- an ArrayLight for the bright
    # patch the sky strikes (the bars and the floor just inside them), and a
    # PointLight, much brighter than the hole's, for the wash down the hall.
    # These belong to the map, not to anything that moves, so they are pinned
    # here like the home level's daylight rather than carried by a portal.
    gx, gy = shift((grate_x, y5))
    xb = shift((x5_e - 2, 0))[0]
    spot = np.zeros(cropped.shape, dtype='float32')
    spot[gy - half:gy + half + 1, xb:gx + 1] = 1.0     # floor just inside the bars
    spot[gy - half - 1:gy + half + 2, gx] = 1.0        # the barred window itself
    sky = np.array([0.8, 0.9, 1.0])
    cropped.add_light(ArrayLight(cropped, spot, color=sky * 100 * lx), pos=(gx, gy))
    cropped.add_light(PointLight(cropped, color=sky * 300 * lm, brightness=1.0), pos=(gx, gy))

    return level, shift(hole_pos), shift(stairs_pos)


def build_dungeon(blocktypes):
    """The original hand-drawn level. Returns (level, arrival_pos).

    The stairs up are stamped where the player used to start, which is where
    the sewer's stairs come out.
    """
    maze = Maze.load_image('level1.png', blocktypes, obj_name='dungeon')
    return Level('dungeon', maze), DUNGEON_ARRIVAL


def build_world(scene):
    """Build the three levels, wire the portals, and populate the map.

    Assembles the home/sewer/dungeon levels, links them, installs the world on
    *scene*, then places everything that belongs to the map: the scroll, the
    torches (at :data:`TORCH_POSITIONS`), the sewer's shaft of daylight, and the
    monster.

    Returns the :class:`~.world.World`, with 'home' current -- it is added
    first, and the first level added is where the game begins.
    """
    world = World()
    bt = world.blocktypes

    home_level, home_hole = build_home(bt)
    sewer_level, sewer_hole, sewer_stairs = build_sewer(bt)
    dungeon_level, dungeon_arrival = build_dungeon(bt)

    world.add_level(home_level)
    world.add_level(sewer_level)
    world.add_level(dungeon_level)

    # The world is live before anything is placed on it, because portal ends,
    # torches and the scroll are all entities that want the scene.
    scene.set_world(world)

    sewer, dungeon = sewer_level.maze, dungeon_level.maze

    # Down the hole and you cannot climb back. The home end is an ordinary hole
    # in the floor -- an 'O' you drop through. The sewer end is the ceiling
    # opening you land under: not enterable (step onto it and it still tells you
    # the way up is out of reach), and invisible (char=None) because there is no
    # hole in the sewer floor -- just the patch of floor the daylight lands on.
    top = Hole(location=(home_level.maze, home_hole), scene=scene)
    bottom = Hole(location=(sewer, sewer_hole), scene=scene,
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
    hx, hy = sewer_hole
    spot = np.zeros(sewer.shape, dtype='float32')
    spot[hy, hx] = 1.0
    bottom.lights = [
        ArrayLight(bottom, spot, color=np.array([0.8, 0.9, 1.0]) * 100*lx),
        PointLight(bottom, color=np.array([0.8, 0.9, 1.0]) * 50*lm, brightness=1.0),
    ]
    world.link(top, bottom)

    # Stairs from the sewer down to the dungeon; each end draws its own glyph.
    world.link(
        StairsDown(location=(sewer, sewer_stairs), scene=scene),
        StairsUp(location=(dungeon, dungeon_arrival), scene=scene)
    )

    world.link(
       StairsDown(location=(home_level.maze, (30, 30)), scene=scene),
       StairsUp(location=(dungeon, (9, 7)), scene=scene)
    )


    Scroll(location=(dungeon, (5, 5)), scene=scene)
    torches = [Torch(location=(dungeon, pos), scene=scene)
               for pos in TORCH_POSITIONS]
    # A brighter, warmer torch than the standard one (Torch.LIGHT_COLOR =
    # (15, 12, 3) lm). Luminous flux per channel in lumens, warm 10:5:1 ratio,
    # ~5x a standard torch so it throws a wider pool. Tunable in the visual pass.
    torches[0].light.color = (75*lm, 37.5*lm, 7.5*lm)

    # The sewer's stairs down get a torch so they are findable from a distance;
    # the ceiling opening the player fell through is lit by its own daylight.
    Torch(location=(sewer, (sewer_stairs[0]-1, sewer_stairs[1])), scene=scene)

    Monster(position=(8, 40), scene=scene, maze=dungeon)

    return world
