"""Level 3, the dungeon: the original hand-drawn level, loaded from an image.

The deepest level, so it wires the portals back to both lower levels: the stairs
up from the sewer, and a shortcut down from home. It also holds most of the
map's content -- the scroll, the torches, and the monster.
"""
from ..item import Scroll, Torch
from ..maze import Maze
from ..monster import Monster
from ..portal import StairsDown, StairsUp
from ..units import lm
from ..world import Level

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


def build_level(scene):
    """Build the dungeon, link it to the sewer and home, and populate it."""
    world = scene.world
    maze = Maze.load_image('level1.png', world.blocktypes, obj_name='dungeon')

    level = Level('dungeon', maze)
    level.locations['sewer_arrival'] = (7, 7)   # where the sewer stairs land
    level.locations['home_arrival'] = (9, 7)    # where home's shortcut lands

    sewer = world.levels['sewer']
    home = world.levels['home']

    # Stairs up from the sewer; each end draws its own glyph. The stairs up are
    # stamped at the dungeon's historical start, which is where the sewer's
    # stairs come out.
    world.link(
        StairsDown(location=(sewer.maze, sewer.locations['stairs_down']), scene=scene),
        StairsUp(location=(maze, level.locations['sewer_arrival']), scene=scene),
    )

    # A shortcut straight down from home.
    world.link(
        StairsDown(location=(home.maze, home.locations['dungeon_stairs']), scene=scene),
        StairsUp(location=(maze, level.locations['home_arrival']), scene=scene),
    )

    Scroll(location=(maze, (5, 5)), scene=scene)
    torches = [Torch(location=(maze, pos), scene=scene)
               for pos in TORCH_POSITIONS]
    # A brighter, warmer torch than the standard one (Torch.LIGHT_COLOR =
    # (15, 12, 3) lm). Luminous flux per channel in lumens, warm 10:5:1 ratio,
    # ~5x a standard torch so it throws a wider pool. Tunable in the visual pass.
    torches[0].light.color = (75 * lm, 37.5 * lm, 7.5 * lm)

    Monster(position=(8, 40), scene=scene, maze=maze)

    return level
