"""The :class:`Lightning` spell.

Game-side module: no rendering library may be imported here.
"""
import time

import numpy as np

from ..units import klm
from .base import Spell


class Lightning(Spell):
    """A bolt of ``/-\\|`` symbols that strobes in 2-3 quick flashes, then dies.

    The bolt is traced :data:`LENGTH` cells from the caster in the chosen
    direction, veering one cell sideways now and then (:data:`WANDER`) so the
    path is almost -- but not quite -- straight. Each cell gets its own ``/-\\|``
    symbol and a fierce cold-white point light. Rather than shine steadily, the
    whole bolt strobes: it snaps on and off :data:`FLASH_RANGE` times, each
    flash lit for :data:`ON_TIME` and dark for :data:`OFF_TIME`, then is torn
    down. Flashing reuses the same lights and sprites -- brightness and sprite
    visibility toggle in place -- so no light churns on and off the level.
    """

    LENGTH = 10
    #: Luminous flux per channel, in lumens (see :class:`~.light.PointLight`).
    #: Fierce cold white: blue >= green > red, and brighter than a fireball.
    COLOR = (15*klm, 18*klm, 27*klm)
    FG = (0.85, 0.92, 1.0, 1.0)
    #: how many quick flashes, picked per cast from ``randint(*FLASH_RANGE)``
    #: (2 or 3); each flash is ON_TIME lit + OFF_TIME dark (whole strobe < 1s).
    FLASH_RANGE = (2, 4)
    ON_TIME = 0.1          # seconds a flash stays lit
    OFF_TIME = 0.1         # seconds of dark between flashes
    WANDER = 0.35           # chance a segment veers one cell sideways
    #: a bolt is far hotter than a fireball -- it superheats the wall it earths
    #: into to blue-white, then cools down through the whole blackbody ramp
    STRIKE_TEMP = 2000.0

    def __init__(self, scene, maze, pos, direction, start=True):
        Spell.__init__(self, scene, maze, entity_type='mob.spell.lightning',
                       obj_name='lightning')
        self._animated = start
        # the wall cell the bolt earths into, if it is stopped by one; set by
        # _trace and turned into a glowing Heat once the lights are built
        self._impact = None
        cells, glyphs = self._trace(np.array(pos, dtype=int),
                                    np.array(direction, dtype=int))
        self.cells = cells

        self.sprite = scene.sprite_layers['actors'].add_sprites((len(cells),))
        self.sprite.glyph = np.array([scene.glyphs[g] for g in glyphs],
                                     dtype='uint32')
        self.sprite.fgcolor = self.FG
        self.sprite.bgcolor = (0, 0, 0, 0)
        # the lit positions, restored on each flash-on; a flash-off hides the
        # sprites (NaN) so the glyphs blink in step with their lights
        self._positions = np.array([(c[0], c[1], -0.15) for c in cells],
                                   dtype='float32')
        self.sprite.position = self._positions

        for c in cells:
            self._add_light(c, self.COLOR)
        self._relight()
        # leave the earthed-into wall glowing (outlives the brief bolt)
        self.heat = self._strike(self._impact) if self._impact is not None else None
        if start:
            self._start()

    def _set_lit(self, on):
        """Show or hide the whole bolt in place (brightness + sprite position).

        Toggling brightness drives each light's ``changed`` signal, so the level
        recomposites and repaints exactly as it does for a flickering flame; the
        sprites are moved off-screen (NaN) while dark so the glyphs blink too.
        """
        if self.sprite is None:
            return
        for light in self.lights:
            light.brightness = 1.0 if on else 0.0
        self.sprite.position = (self._positions if on
                                else np.full_like(self._positions, np.nan))

    def _run(self):
        for _ in range(int(np.random.randint(*self.FLASH_RANGE))):
            self._set_lit(True)
            time.sleep(self.ON_TIME)
            if self._done:
                return
            self._set_lit(False)
            time.sleep(self.OFF_TIME)
            if self._done:
                return
        self.destroy()

    def _trace(self, origin, direction):
        """Return ``(cells, glyphs)`` for an almost-straight bolt from *origin*.

        Steps one cell along *direction* per segment, occasionally adding a
        sideways nudge; each segment's glyph follows the direction it was drawn
        in. Stops early if the path runs into a wall.
        """
        perp = np.array([-direction[1], direction[0]])
        cells = [tuple(int(v) for v in origin)]
        glyphs = [self._glyph(direction)]
        cur = origin.copy()
        for _ in range(self.LENGTH - 1):
            step = direction.copy()
            if np.random.rand() < self.WANDER:
                step = step + perp * np.random.choice([-1, 1])
            nxt = cur + step
            if not self._open(nxt):
                self._impact = tuple(int(v) for v in nxt)
                break
            cells.append(tuple(int(v) for v in nxt))
            glyphs.append(self._glyph(step))
            cur = nxt
        return cells, glyphs

    @staticmethod
    def _glyph(step):
        """The ``/-\\|`` symbol for a one-cell *step*.

        Cardinal steps are ``-``/``|``; a diagonal is ``/`` when it rises to the
        right (dx*dy > 0, with y up) and ``\\`` when it falls to the right.
        """
        dx, dy = int(step[0]), int(step[1])
        if dx and dy:
            return '/' if dx * dy > 0 else '\\'
        return '-' if dx else '|'
