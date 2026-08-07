"""Level 1, home: a small walled room whose only way out is a hole in the floor.

The player starts here. Being the first level, home links to nothing itself --
the sewer hangs its one-way hole from ``locations['hole']`` and the dungeon its
shortcut stairs from ``locations['dungeon_stairs']`` (each level wires the
portals back to lower-numbered levels).
"""
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


def build_level(scene):
    """Build the home level and record its named cells."""
    bt = scene.world.blocktypes
    maze = Maze.filled((40, 120), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')

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

    return level
