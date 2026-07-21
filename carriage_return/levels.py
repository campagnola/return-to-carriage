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
    maze.add_light(AmbientLight(maze, color=(8000, 8000, 8000)), pos=(0, 0))

    return level, hole_pos


def build_sewer(blocktypes, seed=20240719):
    """Four hallways joined end to end, generated from a fixed *seed*.

    Returns ``(level, hole_pos, stairs_pos)``. The hole is at the start of the
    first hallway -- the ceiling opening the player falls in through -- and the
    stairs down are at the far end of the fourth.

    The hallways alternate horizontal and vertical, so each one turns off the
    last, and their lengths come from the seeded generator. Bounds are not left
    to chance: lengths are clamped so a corridor cannot run off the edge, which
    keeps the layout valid for any seed rather than only for this one.
    """
    rng = np.random.RandomState(seed)
    shape = (31, 71)          # rows, cols
    half = 1                  # hallways are 2*half+1 cells wide
    margin = half + 1         # never carve into the outer wall

    maze = Maze.filled(shape, blocktypes, 'wall', obj_name='sewer')
    blocks = maze.blocks
    path = blocktypes.id_of('path')

    x, y = 4, 15
    hole_pos = (x, y)
    blocks[y - half:y + half + 1, x - half:x + half + 1] = path

    for n in range(4):
        if n % 2 == 0:                      # horizontal, always rightward
            room = shape[1] - margin - 1 - x
            length = min(int(rng.randint(12, 19)), room)
            nx = x + length
            blocks[y - half:y + half + 1, x:nx + 1] = path
            x = nx
        else:                               # vertical, up or down
            step = 1 if rng.rand() < 0.5 else -1
            room = (shape[0] - margin - 1 - y) if step > 0 else (y - margin)
            length = min(int(rng.randint(6, 11)), room)
            ny = y + step * length
            lo, hi = min(y, ny), max(y, ny)
            blocks[lo:hi + 1, x - half:x + half + 1] = path
            y = ny

    stairs_pos = (x, y)

    # Crop to the carved area plus a one-cell wall margin. The generator works
    # on a canvas big enough for any seed, but the level that survives should
    # not be mostly solid rock: the sight field is supersampled per maze cell,
    # so unreachable rock costs real memory and time every frame.
    ys, xs = np.nonzero(blocks == path)
    y0, y1 = ys.min() - 1, ys.max() + 2
    x0, x1 = xs.min() - 1, xs.max() + 2
    cropped = Maze(blocks[y0:y1, x0:x1].copy(), blocktypes, obj_name='sewer')
    shift = lambda p: (p[0] - x0, p[1] - y0)
    return Level('sewer', cropped), shift(hole_pos), shift(stairs_pos)


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
    # torch's flame: a bright spot where the shaft strikes the floor (an array
    # light on that one cell), and a dim, shadow-casting 1/r^2 wash for the
    # light that hit the floor there and scattered down the corridor.
    hx, hy = sewer_hole
    spot = np.zeros(sewer.shape, dtype='float32')
    spot[hy, hx] = 6000000.0
    bottom.lights = [
        ArrayLight(bottom, spot, color=(4000, 5000, 8000)),
        PointLight(bottom, color=(10, 20, 100), brightness=1.0),
    ]
    world.link(top, bottom)

    # Stairs from the sewer down to the dungeon; each end draws its own glyph.
    world.link(StairsDown(location=(sewer, sewer_stairs), scene=scene),
               StairsUp(location=(dungeon, dungeon_arrival), scene=scene))

    Scroll(location=(dungeon, (5, 5)), scene=scene)
    torches = [Torch(location=(dungeon, pos), scene=scene)
               for pos in TORCH_POSITIONS]
    torches[0].light.color = (10000, 5000, 1000)

    # The sewer's stairs down get a torch so they are findable from a distance;
    # the ceiling opening the player fell through is lit by its own daylight.
    Torch(location=(sewer, sewer_stairs), scene=scene)

    Monster(position=(8, 40), scene=scene, maze=dungeon)

    return world
