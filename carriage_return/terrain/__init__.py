"""Terrain: shared shape-generating code for the game's outdoor levels --
meandering paths and rivers, patchy grass colour, and ruined buildings.

A level decides *where* these things go -- a path from here to there, a
building near that path -- and terrain decides their exact shape (see
:func:`create_path` and :func:`create_river`, both built on the same
:func:`~.meander.meander` curve) and knows how to paint itself onto a maze.

Game-side package: no rendering library may be imported here.
"""
from .buildings import place_building, try_place_building
from .grass import grass_wash, paint_grass_wash
from .meander import meander
from .path import Path, create_path
from .water import WaterAnimation, WaterBody, create_river

__all__ = [
    'Path', 'WaterAnimation', 'WaterBody',
    'create_path', 'create_river',
    'grass_wash', 'paint_grass_wash',
    'meander',
    'place_building', 'try_place_building',
]
