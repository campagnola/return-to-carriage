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


#: Linear albedo of each slope's straw, before per-strand variation. The ridge
#: runs north-south, so the slopes face west and east. Above 1 in red because
#: home's pinned daylight exposure tone-maps an albedo of 1 well below white;
#: these are chosen for how they read on screen there -- light tan straw on
#: one side, a modestly darker brown on the other.
THATCH_LIGHT_SIDE = (1.15, 0.80, 0.26)
THATCH_DARK_SIDE = (0.62, 0.38, 0.13)

#: Hues individual strands drift toward, a little each: fresher golden straw
#: and older straw weathered grey.
THATCH_GOLDEN = (1.25, 0.90, 0.20)
THATCH_WEATHERED = (0.80, 0.68, 0.42)


def _strand_field(rng, rows, cols, base, res):
    """An ``(rows, cols, 3)`` field of straw strands lying along each row.

    Every texel row is a line of strands laid end to end, each 1-4 cells (of
    *res* texels) long and starting at a random offset, so neighbouring rows'
    strand ends never line up. Each strand has its own brightness and a slight
    drift toward golden or weathered straw; its far tip is darker. Each row
    as a whole is also a little lighter or darker than its neighbours, which
    is what makes the strands read as separate straws rather than bricks.
    """
    out = np.empty((rows, cols, 3), dtype='float32')
    base = np.asarray(base, dtype='float32')
    # the tints are given for the light side; scale them to this side's straw
    scale = base / np.asarray(THATCH_LIGHT_SIDE, dtype='float32')
    golden = np.asarray(THATCH_GOLDEN, dtype='float32') * scale
    weathered = np.asarray(THATCH_WEATHERED, dtype='float32') * scale
    for r in range(rows):
        x = -rng.randint(0, 3 * res)
        row_brightness = np.clip(rng.normal(1.0, 0.12), 0.7, 1.25)
        while x < cols:
            length = int(np.clip(rng.gamma(4.0, res / 2.0), res, 4 * res))
            brightness = row_brightness * np.clip(rng.normal(1.0, 0.2), 0.5, 1.45)
            drift = rng.uniform(-1.0, 1.0)
            tint = golden if drift > 0 else weathered
            color = (base + (tint - base) * abs(drift) * 0.35) * brightness
            lo, hi = max(x, 0), min(x + length, cols)
            if hi > lo:
                out[r, lo:hi] = color
                out[r, hi - 1] *= 0.7
            x += length
    return out


def thatched_roof(building, rng, texels_per_cell=ROOF_TEXELS_PER_CELL):
    """A thatched roof with a north-south ridge down the middle.

    The thatch is straw strands running down each slope -- across the ridge,
    so east-west -- with the west slope light tan (:data:`THATCH_LIGHT_SIDE`)
    and the east slope a modestly darker brown (:data:`THATCH_DARK_SIDE`).
    It is laid in courses parallel to the ridge, each shading faintly darker
    toward its lower edge where the next course's strands overlap it; the
    ridge itself and the roof's outer edges are shaded darker still.
    """
    res = texels_per_cell
    rows, cols = building.h * res, building.w * res
    ridge = cols // 2

    albedo = np.empty((rows, cols, 4), dtype='float32')
    albedo[..., 3] = 1.0
    albedo[:, :ridge, :3] = _strand_field(rng, rows, ridge, THATCH_LIGHT_SIDE, res)
    albedo[:, ridge:, :3] = _strand_field(rng, rows, cols - ridge, THATCH_DARK_SIDE, res)

    # Courses: bands ~1.5 cells wide running parallel to the ridge, measured
    # outward from it, each darkening toward its downslope (outer) edge.
    course = 1.5 * res
    downslope = np.abs(np.arange(cols) + 0.5 - ridge)
    within = (downslope % course) / course
    shade = 1.0 - 0.12 * within ** 2

    # The ridge line and the roof's outer edges (eaves and gable ends).
    shade[max(ridge - 1, 0):ridge + 1] *= 0.72
    shade[[0, -1]] *= 0.75
    albedo[..., :3] *= shade[np.newaxis, :, np.newaxis]
    albedo[[0, -1], :, :3] *= 0.8
    return Roof(building, albedo, res)
