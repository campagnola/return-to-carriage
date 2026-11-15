"""What a monster decides to do on its turn.

A behaviour is kept apart from the :class:`~.base.Monster` body so a kind of
monster can be given, or switch between, different minds without subclassing.
Each exposes one method, ``choose_step(monster, dm) -> (dx, dy) or None``: the
step it would like to take this turn, or None to stay put. It only *chooses*;
the dungeon master still decides whether the step happens.
"""
import random

#: The eight cells around a monster, as ``(dx, dy)`` steps.
STEPS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]


class Idle:
    """Never moves."""

    def choose_step(self, monster, dm):
        return None


class Wander:
    """Step to a random open neighbouring cell each turn; stay put if boxed in.

    *rng* is a :class:`random.Random` (or anything with ``choice``); pass a
    seeded one for a repeatable path.
    """

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()

    def choose_step(self, monster, dm):
        maze, (x, y) = monster.location.place
        open_steps = [(dx, dy) for dx, dy in STEPS
                      if dm.walkable((x + dx, y + dy), maze)]
        if not open_steps:
            return None
        return self.rng.choice(open_steps)
