"""Starting a new game: the player, and the world it drops into.

The world content -- the levels, the portals, the items and torches on them,
the monster -- is built by :func:`.levels.build_world`, so that content is not
mirror-edited across the real game (``return_to_carriage.py``) and the
screenshot regression harness (``agent_helpers/render_screenshot.py``). What is
left here is only what is not part of the map: the player and the torch in its
hand.

Application wiring -- camera following, the command interpreter, input
handlers, the gamepad -- stays with each caller, because the screenshot harness
deliberately omits it (those paths are timing-dependent and would break render
determinism).

Game-side module: it must not import vispy, Qt or OpenGL (see
tests/test_boundaries.py).
"""
from .levels import build_world
from .player import Player
from .item import Torch


def new_game(scene):
    """Start a new game: create the player, build the world,
    and place the player in it.

    Returns ``(world, player)``.
    """
    player = Player(scene)
    world = build_world(scene)
    home = world.levels['home']
    player.location.update(home.maze, home.locations['start'])

    # set player initial inventory

    # held_torch = Torch(location=(player, 'right hand'), scene=scene,
    #                    obj_name="held torch")
    # held_torch.light.color = (10000, 5000, 1000)

    return world, player
