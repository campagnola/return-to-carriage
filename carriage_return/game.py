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

    *torch_positions* is a list of ``(row, col)`` maze cells to place torches
    at; it defaults to :data:`MAIN_TORCH_POSITIONS`. The screenshot harness
    passes a shorter list, because torch count and placement change the
    lighting and therefore the rendered image.

    Returns ``(player, scroll, torches, held_torch, monster)``.
    """
    if torch_positions is None:
        torch_positions = MAIN_TORCH_POSITIONS

    player = Player(scene)
    player.location.update(scene.maze, [7, 7])

    scroll = Scroll(location=(scene.maze, (5, 5)), scene=scene)
    torches = [Torch(location=(scene.maze, pos), scene=scene)
               for pos in torch_positions]
    torches[0].light_color = (10000, 5000, 1000)

    held_torch = Torch(location=(player, 'right hand'), scene=scene,
                       obj_name="held torch")
    held_torch.light_color = (10000, 5000, 1000)

    monster = Monster(position=(8, 40), scene=scene)

    return player, scroll, torches, held_torch, monster
