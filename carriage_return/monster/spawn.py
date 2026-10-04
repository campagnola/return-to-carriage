"""Spawning monsters on the player's level as the player moves around.

A :class:`MonsterSpawner` listens to the dungeon master's ``turn_ended`` and,
every :data:`SPAWN_INTERVAL` turns (re-rolled after each spawn), puts a new
wandering :class:`~.letter.Letter` on the level the player is on: at a random
floor cell, reachable from the player, at least :data:`MIN_SPAWN_DISTANCE` moves
away and with nothing already standing on it. "Moves" are horizontal and vertical
steps over walkable terrain, so a cell just the other side of a wall is as far
away as the walk around it.

Game-side module: no rendering library may be imported here.
"""
import random

import numpy as np

from .behaviour import Wander
from .letter import LETTERS, Letter

#: Inclusive range of player turns between one spawn and the next.
SPAWN_INTERVAL = (50, 80)

#: Fewest horizontal/vertical moves from the player a monster may spawn at.
MIN_SPAWN_DISTANCE = 10


def move_distances(walkable, start):
    """Fewest horizontal/vertical moves from *start* ``(x, y)`` to every cell.

    *walkable* is a ``(h, w)`` bool array; moves only cross walkable cells.
    Returns a ``(h, w)`` int array, ``-1`` where a cell cannot be reached. A
    breadth-first search grown one ring of moves per step over the whole grid.
    """
    h, w = walkable.shape
    dist = np.full((h, w), -1, dtype=int)
    x, y = start
    dist[y, x] = 0
    frontier = np.zeros((h, w), dtype=bool)
    frontier[y, x] = True
    d = 0
    while frontier.any():
        d += 1
        grown = np.zeros_like(frontier)
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        frontier = grown & walkable & (dist < 0)
        dist[frontier] = d
    return dist


class MonsterSpawner:
    """Spawns a letter monster near-ish the player every few turns.

    *rng* is a :class:`random.Random`; pass a seeded one for repeatable spawns.
    Connect it to a dungeon master with :meth:`attach`.
    """

    def __init__(self, scene, rng=None):
        self.scene = scene
        self.rng = rng if rng is not None else random.Random()
        #: turns left until the next spawn is due; at zero or below a spawn is
        #: attempted every turn until one succeeds
        self.countdown = self._roll_interval()

    def _roll_interval(self):
        return self.rng.randint(*SPAWN_INTERVAL)

    def attach(self, dm):
        dm.turn_ended.connect(self.on_turn)

    def on_turn(self, dm):
        """A player turn ended: count it down and spawn if one is due."""
        self.countdown -= 1
        if self.countdown <= 0 and self.spawn() is not None:
            self.countdown = self._roll_interval()

    def spawn_cells(self):
        """``(n, 2)`` array of the ``(x, y)`` cells a monster may spawn at now."""
        player = self.scene.player
        ml = player.location.global_location if player is not None else None
        if ml is None:
            return np.empty((0, 2), dtype=int)
        maze, start = ml.container, ml.slot
        walkable = maze.blocktypes['walkable'][maze.blocks].astype(bool)
        ok = move_distances(walkable, start) >= MIN_SPAWN_DISTANCE
        for entity in maze.inventory.all_entities():
            x, y = entity.location.slot
            ok[y, x] = False
        ys, xs = np.nonzero(ok)
        return np.stack([xs, ys], axis=1)

    def spawn(self):
        """Put a wandering letter on a random spawn cell of the player's level.

        Returns the new monster, or None if no cell qualifies.
        """
        cells = self.spawn_cells()
        if len(cells) == 0:
            return None
        x, y = cells[self.rng.randrange(len(cells))]
        maze = self.scene.player.location.global_location.container
        return Letter(self.rng.choice(LETTERS), location=(maze, (int(x), int(y))),
                      scene=self.scene, behaviour=Wander(self.rng))
