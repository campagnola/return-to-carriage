"""Shared world setup.

Both the real game (``return_to_carriage.py``) and the screenshot regression
harness (``agent_helpers/render_screenshot.py``) need the same world content:
the player start, the scroll, the torches, the held torch and the monster.
This module holds that content in one place so world construction is not
mirror-edited across two files.

Only *world content* lives here. Application wiring -- camera following, the
command interpreter, input handlers, the gamepad -- stays with each caller,
because the screenshot harness deliberately omits it (those paths are
timing-dependent and would break render determinism).

Game-side module: it must not import vispy, Qt or OpenGL (see
tests/test_boundaries.py).
"""
from .levels import HOME_START, build_world
from .player import Player
from .monster import Monster
from .item import Scroll, Torch


#: Torch positions used by the main game.
MAIN_TORCH_POSITIONS = [
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


def new_game(scene, torch_positions=None):
    """Populate *scene* with the starting world content.

    Builds the three-level world (home, sewer, dungeon), installs it on the
    scene, and places the player, the scroll, the torches and the monster.
    The player starts on the home level and reaches the dungeon by falling
    down the hole and taking the sewer stairs.

    *torch_positions* is a list of ``(row, col)`` dungeon cells to place
    torches at; it defaults to :data:`MAIN_TORCH_POSITIONS`. The screenshot
    harness passes a shorter list, because torch count and placement change
    the lighting and therefore the rendered image.

    Returns ``(player, scroll, torches, held_torch, monster)``.
    """
    if torch_positions is None:
        torch_positions = MAIN_TORCH_POSITIONS

    world = build_world()
    scene.set_world(world)
    home = world.levels['home'].maze
    sewer = world.levels['sewer'].maze
    dungeon = world.levels['dungeon'].maze

    player = Player(scene)
    player.location.update(home, HOME_START)

    scroll = Scroll(location=(dungeon, (5, 5)), scene=scene)
    torches = [Torch(location=(dungeon, pos), scene=scene)
               for pos in torch_positions]
    torches[0].light_color = (10000, 5000, 1000)

    # The sewer is lit at its two landmarks: the shaft the player falls down
    # and the stairs at the far end, so each is findable from a distance.
    for end in world.portals[0].ends + world.portals[1].ends:
        if end.level.maze is sewer:
            torches.append(Torch(location=(sewer, end.pos), scene=scene))

    held_torch = Torch(location=(player, 'right hand'), scene=scene,
                       obj_name="held torch")
    held_torch.light_color = (10000, 5000, 1000)

    monster = Monster(position=(8, 40), scene=scene, maze=dungeon)

    return player, scroll, torches, held_torch, monster
