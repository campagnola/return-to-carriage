"""Level 1, home: a small walled room whose only way out is a hole in the floor.

The player starts here. Being the first level, home links to nothing itself --
the sewer hangs its one-way hole from ``locations['hole']`` and the dungeon its
shortcut stairs from ``locations['dungeon_stairs']`` (each level wires the
portals back to lower-numbered levels).
"""
import numpy as np

from .. import terrain
from ..light import AmbientLight
from ..maze import Maze
from ..world import Level


#: The luminance home holds the eye fixed at, in cd/m^2. Home is even daylight,
#: so how bright it should *feel* is a property of the sky pouring in, not of the
#: dim floor the adaptation window happens to sample -- and that sample drifts as
#: the player moves between open grass and the brighter walls, which reads as the
#: room dimming and brightening for no reason. Pinning the eye here (both bounds
#: equal, see EyeAdaptation.set_bounds) fixes the exposure to steady daylight.
#: Chosen at the reflected luminance of the sunlit grass floor (its albedo under
#: the 50000 lx wash below); raise it toward MAX_ADAPT_LUMINANCE to make home
#: read brighter. Tunable in the visual pass.
HOME_ADAPT_LUMINANCE = 5000.0

#: 'start', 'hole' and 'dungeon_stairs' all sit in the top-left corner (x<31,
#: y<31); the town is kept well clear of that corner so it never overwrites them.
PATH_WIDTH = 2


def _paint_town(maze, bt, seed=None, start=True):
    """Paint a river, two dirt paths, a bridge, a handful of ruined buildings,
    and patchy grass colour into home's interior. Returns the river's
    :class:`~..terrain.water.WaterBody`.

    This is home's own placement logic -- where the river runs, where the
    paths run and where they meet, where the bridge crosses, where buildings
    cluster -- while :mod:`~..terrain` decides the exact shape each of those
    takes and how it paints onto the maze.

    Everything is generated from *seed* (``None`` draws fresh entropy, so the
    layout differs every game; tests pass an explicit seed for determinism).
    Fixed points elsewhere in the level -- 'start', 'hole', 'dungeon_stairs' --
    all sit in the top-left corner, so the town is based well away from it.
    *start* controls whether the river's WaterAnimation starts its animation
    thread immediately (see terrain.create_river); tests that don't need it
    running pass ``start=False``.
    """
    rng = np.random.RandomState(seed)
    blocks = maze.blocks
    rows, cols = blocks.shape
    x_lo, x_hi = 1, cols - 2
    y_lo, y_hi = 1, rows - 2

    dirt_id = bt.id_of('dirt')
    bridge_id = bt.id_of('bridge')

    # The river: north/south, meandering in x, on the east side of the room
    # -- far from the start/hole/dungeon_stairs corner.
    river_cx = rng.randint(x_lo + int(cols * 0.55), x_lo + int(cols * 0.72))
    # animate=False: the bridge below still has to draw over part of the
    # river, and the animation's display mask must see that final state --
    # see terrain.create_river.
    river = terrain.create_river(
        maze, bt.id_of('river'), rng, (river_cx, y_lo), (river_cx, y_hi),
        amplitude=int(cols * 0.06), wavelength=130, bounds=(x_lo, x_hi), animate=False)

    # The east-west path: meanders in y, crossing the whole room (and, on the
    # east side, the river).
    path1_cy = rng.randint(y_lo + int(rows * 0.35), y_lo + int(rows * 0.60))
    path1 = terrain.create_path(
        rng, (x_lo, path1_cy), (x_hi, path1_cy), amplitude=int(rows * 0.10),
        wavelength=145, width=PATH_WIDTH, blocktype_id=dirt_id, bounds=(y_lo, y_hi))
    path1.paint(maze)

    # The north-south path: meanders in x, starting at the north wall and
    # running only until it meets the east-west path. Built over the room's
    # full height first so the crossing can be found, then trimmed to it.
    path2_cx = rng.randint(x_lo + int(cols * 0.15), x_lo + int(cols * 0.32))
    path2_full = terrain.create_path(
        rng, (path2_cx, y_lo), (path2_cx, y_hi), amplitude=int(cols * 0.03),
        wavelength=25, width=PATH_WIDTH, blocktype_id=dirt_id, bounds=(x_lo, x_hi))
    ys = np.arange(y_lo, y_hi + 1)
    crossing_y = path1.centerline[path2_full.centerline - path1.lo]
    intersection_y = ys[np.argmin(np.abs(crossing_y - ys))]
    path2 = path2_full.trimmed(intersection_y)
    path2.paint(maze)

    # The bridge: a simple rectangle where the east-west path crosses the
    # river -- as tall as path1's own band (so it doesn't spill outside the
    # path), and as long as the river's width at that band plus one block of
    # margin on each side. Found from where path1's mask and the river's
    # actually overlap (a small, local patch of cells); the crossing's
    # representative x is the middle of that patch. A full-column scan for
    # the river's extent (any row it ever occupies across the whole room) is
    # wrong here: the river is mean-reverting, so it lingers near the same x
    # for long stretches, which would pick up nearly every row rather than
    # just the crossing -- hence restricting the river's column extent to
    # path1's own row-band at the crossing.
    path1_mask = path1.mask(blocks.shape)
    overlap_rows, overlap_cols = np.nonzero(path1_mask & river.mask)
    bridge_x = int(np.median(overlap_cols))
    by_lo, by_hi = path1.band_at(bridge_x)
    river_cols = np.nonzero(river.mask[by_lo:by_hi, :].any(axis=0))[0]
    bx_lo = max(river_cols.min() - 1, x_lo)
    bx_hi = min(river_cols.max() + 1, x_hi)
    blocks[by_lo:by_hi, bx_lo:bx_hi + 1] = bridge_id

    # Now that the bridge has drawn over its stretch of the river, the
    # animation can snapshot which cells are still actually showing as river.
    river.animate(maze, seed=seed, start=start)

    # Town centre: the stretch of the east-west path between the path
    # intersection and the bridge. A handful of ruined buildings line it,
    # plus a couple more scattered nearby.
    town_x_lo, town_x_hi = sorted((path2_full.center_at(intersection_y), bridge_x))
    b_bounds = (x_lo, x_hi), (y_lo, y_hi)
    n_along = rng.randint(2, 5)
    for x in np.clip(
            np.linspace(town_x_lo, town_x_hi, n_along + 2)[1:-1]
            + rng.uniform(-6, 6, size=n_along),
            town_x_lo, town_x_hi).astype(int):
        side = rng.choice((-1, 1))
        cx, cy = path1.point_at(x, side * rng.randint(6, 11))
        terrain.try_place_building(blocks, bt, rng, cx, cy, *b_bounds)

    town_cx = (town_x_lo + town_x_hi) // 2
    town_cy = path1.center_at(town_cx)
    for _ in range(2):
        radius = rng.randint(12, 24)
        angle = rng.uniform(0, 2 * np.pi)
        cx = town_cx + int(radius * np.cos(angle))
        cy = town_cy + int(radius * np.sin(angle))
        terrain.try_place_building(blocks, bt, rng, cx, cy, *b_bounds)

    # Patchy grass colour, mixed lightly into the grass floor everywhere it
    # still shows -- painted last so it follows the final grass footprint.
    terrain.paint_grass_wash(maze, bt, rng)

    return river


def build_level(scene):
    """Build the home level and record its named cells."""
    bt = scene.world.blocktypes
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river = _paint_town(maze, bt)

    level = Level('home', maze)
    level.locations['start'] = (3, 5)             # where the player begins
    level.locations['hole'] = (11, 5)             # the sewer's hole lands here
    level.locations['dungeon_stairs'] = (30, 30)  # a shortcut down to the dungeon

    # Home holds the eye at a fixed daylight exposure rather than sampling its
    # floor (see HOME_ADAPT_LUMINANCE). Equal bounds pin it; leaving the eye
    # daylight-adapted here is what makes the first moment down the hole dark.
    level.min_adapt_luminance = level.max_adapt_luminance = HOME_ADAPT_LUMINANCE

    # The even wash of daylight through the roof. A map light: it belongs to the
    # room, not to anything that moves, and needs no scene -- it announces any
    # change through its own signal, which the level handles. A map light only
    # registers with a level that already exists (see Light._register), so it is
    # added after the level is built.
    maze.add_light(AmbientLight(maze, color=(50000,) * 3), pos=(0, 0))

    # The river's animated shimmer, built by create_river inside _paint_town.
    # Kept on the maze rather than discarded so it isn't garbage collected out
    # from under its own thread and so tests/tools can reach it (e.g. to call
    # .advance() without sleeping).
    maze.water_animation = river.animation

    return level
