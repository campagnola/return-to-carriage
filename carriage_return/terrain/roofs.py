"""Roofs: an overlay covering a building's whole footprint, seen from above,
that turns transparent while the player is inside so the floor and walls
beneath show through.

A roof is game state: where it sits (its :class:`~.buildings.Building`), what
it looks like (an albedo texture), and three per-frame facts the level keeps
current in :meth:`~..world.Level.update_sight` -- whether the player is under
it, how much of it is in view, and how brightly it is remembered. How it is
drawn, and how quickly it fades between covered and open, is the renderer's
business.

Game-side module: no rendering library may be imported here.
"""
import numpy as np

from ..tone_mapping import LUMINANCE_WEIGHTS


#: Albedo texels per maze cell along each axis. Fine enough for detail
#: narrower than a glyph.
ROOF_TEXELS_PER_CELL = 8


class Roof:
    """One building's roof.

    ``albedo`` is a float32 ``(h * res, w * res, 4)`` RGBA texture over the
    building's footprint, ``res`` = :attr:`texels_per_cell`; row 0 is the
    footprint's first maze row (``building.y0``), column 0 its first column.
    rgb is linear reflectance, lit and tone-mapped like any other surface;
    alpha is coverage.

    State written each frame by :meth:`update_sight`:

    - ``open``: the player is standing within the footprint, so the roof
      should be drawn transparent.
    - ``seen``: line of sight to the building, 0..1 -- the most of any texel
      of its footprint. Walls are what a viewer outside actually sees, and
      seeing a building's walls means seeing its roof.
    - ``remembered``: display-space memory of the building -- the brightest
      remembered wall face in its footprint (see
      :mod:`~..sight_memory`), so the roof fades from memory exactly as its
      walls do.
    """

    def __init__(self, building, albedo, texels_per_cell=ROOF_TEXELS_PER_CELL):
        res = texels_per_cell
        expected = (building.h * res, building.w * res, 4)
        if albedo.shape != expected:
            raise ValueError('albedo shape %r, expected %r' % (albedo.shape, expected))
        self.building = building
        self.texels_per_cell = res
        self.albedo = np.ascontiguousarray(albedo, dtype='float32')
        self.open = False
        self.seen = 0.0
        self.remembered = 0.0

    @property
    def albedo_luminance(self):
        """Mean luminance of the roof's covered texels; the renderer scales the
        remembered roof by ``albedo / albedo_luminance`` so its texture
        survives into memory at the walls' remembered brightness."""
        a = self.albedo
        lum = a[..., :3] @ LUMINANCE_WEIGHTS
        covered = a[..., 3] > 0
        return float(lum[covered].mean()) if covered.any() else 1.0

    def update_sight(self, player_pos, los_scalar, memory, supersample):
        """Refresh ``open``/``seen``/``remembered`` for this frame.

        *player_pos* is the player's ``(x, y)`` cell, or None when the player
        is not on this roof's level. *los_scalar* and *memory* are the level's
        ``(h, w)`` line-of-sight and memory fields at *supersample* texels per
        cell.
        """
        footprint = self.building.field_slice(supersample)
        if player_pos is None:
            self.open = False
            self.seen = 0.0
        else:
            self.open = self.building.contains(player_pos)
            self.seen = float(los_scalar[footprint].max())
        self.remembered = float(memory[footprint].max())

    def __repr__(self):
        return 'Roof(%r)' % (self.building,)


def plain_roof(building, color, texels_per_cell=ROOF_TEXELS_PER_CELL):
    """A roof of one solid, opaque *color* (linear rgb albedo)."""
    res = texels_per_cell
    albedo = np.empty((building.h * res, building.w * res, 4), dtype='float32')
    albedo[..., :3] = color
    albedo[..., 3] = 1.0
    return Roof(building, albedo, res)
