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


def paint_town(maze, bt, seed=None, start=True):
    """Paint a river, two dirt paths, a bridge, a handful of ruined buildings,
    sandy river banks, lush riverside greenery, and patchy grass colour into
    home's interior. Returns ``(river, town_center)``: the river's
    :class:`~..terrain.water.WaterBody` (with its banks attached as ``.banks``
    and greenery as ``.greenery``, see :func:`~..terrain.water.paint_river_banks`
    and :func:`~..terrain.water.paint_river_greenery`), and the ``(x, y)`` cell
    at the middle of the town.

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
        maze, bt.id_of('river'), rng, (river_cx, y_hi), (river_cx, y_lo), width=12,
        amplitude=int(cols * 0.06), wavelength=130, bounds=(x_lo, x_hi), animate=False,
        bed_albedo=(.3, .2, .05))

    # The east-west path: meanders in y on each side of the river, but is
    # built as two separate halves that both end exactly at crossing_y --
    # the row the bridge below sits on -- so the path runs dead straight
    # where it actually meets the river, rather than however a single
    # meander across the whole room happened to be wandering at that point.
    crossing_y = rng.randint(y_lo + int(rows * 0.35), y_lo + int(rows * 0.60))
    half = PATH_WIDTH // 2
    band_lo, band_hi = crossing_y - half, crossing_y - half + PATH_WIDTH
    river_cols = np.nonzero(river.mask[band_lo:band_hi, :].any(axis=0))[0]
    bridge_x_lo = max(river_cols.min() - 1, x_lo)
    bridge_x_hi = min(river_cols.max() + 1, x_hi)

    # meander() guarantees both ends of the centerline land exactly on the
    # coordinates given here (see its Brownian-bridge correction), so the
    # cell touching the bridge on each side is exactly crossing_y with no
    # further nudging needed.
    path1_west = terrain.create_path(
        rng, (x_lo, crossing_y), (bridge_x_lo - 1, crossing_y), amplitude=int(rows * 0.10),
        wavelength=145, width=PATH_WIDTH, blocktype_id=dirt_id, bounds=(y_lo, y_hi))
    path1_west.paint(maze)

    path1_east = terrain.create_path(
        rng, (bridge_x_hi + 1, crossing_y), (x_hi, crossing_y), amplitude=int(rows * 0.10),
        wavelength=145, width=PATH_WIDTH, blocktype_id=dirt_id, bounds=(y_lo, y_hi))
    path1_east.paint(maze)

    # The north-south path: meanders in x, starting at the north wall and
    # running only until it meets the east-west path's western half. Built
    # over the room's full height first so the crossing can be found, then
    # trimmed to it.
    path2_cx = rng.randint(x_lo + int(cols * 0.15), x_lo + int(cols * 0.32))
    path2_full = terrain.create_path(
        rng, (path2_cx, y_lo), (path2_cx, y_hi), amplitude=int(cols * 0.03),
        wavelength=25, width=PATH_WIDTH, blocktype_id=dirt_id, bounds=(x_lo, x_hi))
    ys = np.arange(y_lo, y_hi + 1)
    path1_y_at_path2_x = path1_west.centerline[path2_full.centerline - path1_west.lo]
    intersection_y = ys[np.argmin(np.abs(path1_y_at_path2_x - ys))]
    path2 = path2_full.trimmed(intersection_y)
    path2.paint(maze)

    # The bridge: a simple rectangle spanning the gap left between the two
    # path halves above -- as tall as their shared band at crossing_y, and
    # as wide as the river's extent in that band (already found above, with
    # one block of margin on each side, when sizing where the path halves
    # stop).
    blocks[band_lo:band_hi, bridge_x_lo:bridge_x_hi + 1] = bridge_id

    # Now that the bridge has drawn over its stretch of the river, the
    # animation can snapshot which cells are still actually showing as river.
    # river.animate(maze, seed=seed, start=start)

    # Town centre: the stretch of the east-west path between the path
    # intersection and the bridge. A handful of ruined buildings line it,
    # plus a couple more scattered nearby.
    town_x_lo, town_x_hi = sorted((path2_full.center_at(intersection_y), path1_west.hi))
    b_bounds = (x_lo, x_hi), (y_lo, y_hi)
    n_along = rng.randint(2, 5)
    for x in np.clip(
            np.linspace(town_x_lo, town_x_hi, n_along + 2)[1:-1]
            + rng.uniform(-6, 6, size=n_along),
            town_x_lo, town_x_hi).astype(int):
        side = rng.choice((-1, 1))
        cx, cy = path1_west.point_at(x, side * rng.randint(6, 11))
        terrain.try_place_building(blocks, bt, rng, cx, cy, *b_bounds)

    town_cx = (town_x_lo + town_x_hi) // 2
    town_cy = path1_west.center_at(town_cx)
    town_center = (town_cx, town_cy)
    for _ in range(2):
        radius = rng.randint(12, 24)
        angle = rng.uniform(0, 2 * np.pi)
        cx = town_cx + int(radius * np.cos(angle))
        cy = town_cy + int(radius * np.sin(angle))
        terrain.try_place_building(blocks, bt, rng, cx, cy, *b_bounds)

    # Sandy banks: 0-1 blocks of sand just outside the river's edges, over
    # whatever grass is still exposed now that paths, the bridge and
    # buildings have all claimed their own footprint. Kept on the river
    # itself (river.banks) so the greenery pass below can start just past it.
    river.banks = terrain.paint_river_banks(maze, bt, river, rng)

    # Lush greenery: 0-3 blocks of a deep, saturated green wash just past the
    # banks, fading back into ordinary grass -- built directly on river.banks
    # so it starts exactly where the actually-placed sand ends.
    river.greenery = terrain.paint_river_greenery(maze, bt, river, river.banks, rng)

    # Patchy grass colour, mixed lightly into the grass floor everywhere it
    # still shows -- painted last so it composes on top of the greenery wash
    # above (washes layer, see Maze.wash_bg_color) rather than replacing it.
    terrain.paint_grass_wash(maze, bt, rng)

    return river, town_center


def build_level(scene):
    """Build the home level and record its named cells."""
    bt = scene.world.blocktypes
    maze = Maze.filled((100, 300), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')
    river, town_center = paint_town(maze, bt)

    level = Level('home', maze)
    level.locations['start'] = (3, 5)             # where the player begins
    level.locations['hole'] = (11, 5)             # the sewer's hole lands here
    level.locations['dungeon_stairs'] = (30, 30)  # a shortcut down to the dungeon
    level.locations['town'] = town_center         # the town centre, by the bridge

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
