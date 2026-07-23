"""Builders for the levels the game ships with, and the world that joins them.

One module per level, named ``level_NNN_short_desc`` so the files sort into
build order and read as a table of contents. Each exposes a single entry point,
``build_level(scene) -> Level``, which builds the level, populates whatever
belongs to its map (torches, the scroll, shafts of daylight, the monster) and
wires the portals between itself and any *lower*-numbered level. Because levels
are built in numerical order, a level only ever links back to levels that
already exist, so every portal is owned by exactly one builder.

The coordinates a portal hangs from live in each level's ``locations`` dict (see
:class:`~.world.Level`): a builder records its own named cells there, and a
higher-numbered builder reads them when it links back.

:func:`build_world` discovers the level modules, builds them in order, and
assembles the world -- ``home`` first, so the game begins there. All map content
lives in the level modules so it is not mirror-edited across the game and the
screenshot harness. Only the player and its inventory are placed by the caller
(see :func:`.game.new_game`).

Game-side package: no rendering library may be imported here.
"""
import importlib
import pkgutil

from ..world import World


def _ordered_builders():
    """Yield each level module's ``build_level``, in numerical order.

    The ``level_NNN_*`` naming makes lexical order the same as numerical order,
    so new levels drop in without touching this loader.
    """
    names = sorted(name for _, name, _ in pkgutil.iter_modules(__path__)
                   if name.startswith('level_'))
    for name in names:
        module = importlib.import_module('.' + name, __package__)
        yield module.build_level


def build_world(scene):
    """Build every shipped level in order, wire the portals, and populate the map.

    Returns the :class:`~.world.World`, with the first level current -- it is
    added first, and the first level added is where the game begins.

    The world is made live on *scene* before the first level is built: each
    ``build_level`` reads the shared blocktype table off ``scene.world`` and
    looks up the lower-numbered levels it links back to. :meth:`Scene.set_world`
    needs a current level, which does not exist yet, so the world is attached
    here directly and the first level shown as soon as it has been added.
    """
    world = World()
    scene.world = world
    for index, build_level in enumerate(_ordered_builders()):
        level = build_level(scene)
        world.add_level(level)
        if index == 0:
            scene.set_level(level)
    return world
