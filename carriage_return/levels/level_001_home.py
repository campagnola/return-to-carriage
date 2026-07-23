"""Level 1, home: a small walled room whose only way out is a hole in the floor.

The player starts here. Being the first level, home links to nothing itself --
the sewer hangs its one-way hole from ``locations['hole']`` and the dungeon its
shortcut stairs from ``locations['dungeon_stairs']`` (each level wires the
portals back to lower-numbered levels).
"""
from ..light import AmbientLight
from ..maze import Maze
from ..world import Level


def build_level(scene):
    """Build the home level and record its named cells."""
    bt = scene.world.blocktypes
    maze = Maze.filled((40, 120), bt, 'wall', obj_name='home')
    maze.blocks[1:-1, 1:-1] = bt.id_of('grass')

    level = Level('home', maze)
    level.locations['start'] = (3, 5)             # where the player begins
    level.locations['hole'] = (11, 5)             # the sewer's hole lands here
    level.locations['dungeon_stairs'] = (30, 30)  # a shortcut down to the dungeon

    # The even wash of daylight through the roof. A map light: it belongs to the
    # room, not to anything that moves, and needs no scene -- it announces any
    # change through its own signal, which the level handles. A map light only
    # registers with a level that already exists (see Light._register), so it is
    # added after the level is built.
    maze.add_light(AmbientLight(maze, color=(50000,) * 3), pos=(0, 0))

    return level
