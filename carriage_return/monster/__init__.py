"""Monsters: turn-taking mobs that stand in the maze.

A :class:`Monster` is an :class:`~.entity.Entity` with a location in a maze's
inventory, so walkability, ``on_walked_on`` and the cell lookups all see it like
anything else standing there. What it does each turn is decided by its
*behaviour* (see :mod:`.behaviour`); whether that actually happens is decided by
the dungeon master, exactly as for the player.

Game-side package: no rendering library may be imported here.
"""
from .base import Monster
from .behaviour import Idle, Wander
