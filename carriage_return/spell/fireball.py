"""The :class:`Fireball` spell.

Game-side module: no rendering library may be imported here.
"""
import time

import numpy as np

from ..heat import blackbody_color
from .base import Spell


class Fireball(Spell):
    """A blazing point light that flies ``*`` in a straight line until a wall.

    One very bright, slightly-white point light (about ten times a torch, its
    blue lifted so the core reads warm-white rather than orange) rides a single
    ``*`` sprite. It advances at :data:`SPEED` cells/second, one cell at a time
    so a wall is never stepped over, and snuffs itself the instant the next cell
    is not walkable.
    """

    GLYPH = '*'
    #: Luminous flux per channel, in lumens (see :class:`~.light.PointLight`).
    #: ~10x a torch (``Torch.LIGHT_COLOR`` = (15, 12, 3)) with the blue channel
    #: lifted so the fierce core reads warm-white, not orange.
    COLOR = np.array(blackbody_color(1500)) * 15000.0
    FG = (1.0, 0.95, 0.8, 1.0)
    SPEED = 50.0            # cells / second
    STEP_INTERVAL = 1 / 60.
    #: the wall it hits glows fresh-ember hot (a full-brightness orange glow)
    STRIKE_TEMP = 1000.0

    def __init__(self, scene, maze, pos, direction, start=True):
        Spell.__init__(self, scene, maze, entity_type='mob.spell.fireball',
                       obj_name='fireball')
        self._animated = start
        self.heat = None
        self.direction = np.array(direction, dtype=int)
        self.pos = np.array(pos, dtype=int)
        self._float_pos = self.pos.astype(float)

        self.sprite = scene.sprite_layers['actors'].add_sprites((1,))
        self.sprite.glyph = scene.glyphs[self.GLYPH]
        self.sprite.fgcolor = self.FG
        self.sprite.bgcolor = (0, 0, 0, 0)

        self._add_light(self.pos, self.COLOR)
        self._draw()
        self._relight()
        if start:
            self._start()

    def _draw(self):
        x, y = int(self.pos[0]), int(self.pos[1])
        # zval below the player (-0.2) but above items/mobs (-0.1) so the bolt
        # reads clearly as it passes over them
        self.sprite.position = (float(x), float(y), -0.15)

    def _advance_to(self, pos):
        self.pos = pos
        self.lights[0].pin(self.maze, tuple(int(v) for v in pos))
        self._draw()
        self._relight()

    def advance(self, dt):
        """Fly the bolt forward *dt* seconds; snuff it on a wall.

        Walks one cell at a time toward the accumulated float position so a
        fast bolt on a long frame cannot leap over a wall. Returns False (and
        destroys the bolt) the moment the next cell is not walkable, True while
        it still lives. Injectable ``dt`` so tests drive travel without sleeping.
        """
        self._float_pos += self.direction * self.SPEED * dt
        while not self._done:
            step = np.clip(np.trunc(self._float_pos - self.pos), -1, 1).astype(int)
            if not step.any():
                break
            nxt = self.pos + step
            if not self._open(nxt):
                self.heat = self._strike(nxt)   # leave the wall glowing
                self.destroy()
                return False
            self._advance_to(nxt)
        return not self._done

    def _run(self):
        last = time.perf_counter()
        while not self._done:
            time.sleep(self.STEP_INTERVAL)
            now = time.perf_counter()
            dt, last = now - last, now
            self.advance(dt)
