# coding: utf8
"""Items that live on the map: a base entity plus one module per item.

An :class:`Item` is an :class:`~.entity.Entity` that sits in the maze
inventory, can be looked at, and may be taken into a player's own inventory.
:mod:`.base` holds the shared plumbing; each concrete item gets its own module:

- :class:`Scroll` -- a readable scroll whose pages send the reader in circles.
- :class:`Sword` -- a rusty sword you notice differently by light.
- :class:`Torch` -- a burning, flickering light source (see :mod:`.light`).

Game-side package: no rendering library may be imported here.
"""
from .base import Item
from .scroll import Scroll
from .sword import Sword
from .torch import Torch
