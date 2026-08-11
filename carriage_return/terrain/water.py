"""Water: shape, flow and the shimmering animation of a body of water.

A level builds its water with a ``create_*`` function (currently just
:func:`create_river`; a future :func:`create_pond` or :func:`create_whirlpool`
would join it) -- each responsible for its own shape and flow, and each
returning a :class:`WaterBody`, the generic baked shape/flow data that paints
itself onto a maze. :class:`WaterAnimation` then animates a WaterBody: a
per-cell "water noise" value, advected downstream along the body's flow
direction and remapped through a deep-blue-to-white colour ramp, refreshed on
its own daemon thread in real time -- the same pattern as Heat and Torch.

Game-side module: no rendering library may be imported here.
"""
import threading
import time

import numpy as np
import scipy.sparse

from .meander import meander, natural_extent
from .path import create_path
from ..glyph_effects import ColorModifier


#: Colour ramp water noise is mapped through (see _restyle). The update in
#: advance() damps noise heavily toward its mean each step (SELF_WEIGHT +
#: NEIGHBOR_WEIGHT = 0.9), so nearly every cell sits in a narrow band around
#: the steady state rather than spanning 0..1 -- see the module docstring's
#: note on RANDOM_SKEW. A plain two-colour lerp would then paint almost the
#: whole river the one blended colour, which lands as a flat grey-white
#: rather than blue. Spreading three stops before white instead keeps most of
#: the plausible range solidly blue and reserves foam for genuine excursions.
WATER_COLOR_RAMP_T = (0.0, 0.45, 0.75, 1.0)
WATER_COLOR_RAMP_RGB = (
    (0.02, 0.06, 0.20),   # still, deep water
    (0.05, 0.22, 0.50),   # the river's typical, saturated blue
    (0.30, 0.50, 0.72),   # choppier water
    (0.88, 0.93, 1.00),   # white foam, only at the very top of the range
)


#: Riverbed albedo -- a lit, sandy/silty tone, what the bed itself looks like
#: with no water tinting at all. Shows through almost undiminished at the
#: water's edge (depth_fraction == 0) and fades out toward the centreline.
#: Deliberately close to what a later step's sandy banks will use, since a
#: riverbed is really just submerged bank material -- but that's not this
#: step's concern; retune independently if the two end up clashing.
RIVER_BED_ALBEDO = (0.55, 0.47, 0.32)

#: What the water reads as once the bed's contribution has fully vanished
#: (depth_fraction -> 1): a cousin of WATER_COLOR_RAMP_RGB's "still, deep
#: water" and "typical blue" stops, sitting between the two, so the animated
#: shimmer layered on top (see create_river / WaterAnimation) reads as the
#: same river rather than a second, unrelated one.
RIVER_DEEP_WATER_COLOR = (0.03, 0.10, 0.38)

#: Per-channel Beer-Lambert absorption coefficients (per metre) for
#: water_bed_color's exponential transmission (T = exp(-k * depth)): red
#: highest so it attenuates fastest (matching real water, which absorbs red
#: light quickest), blue lowest so it penetrates deepest -- the shallows read
#: warm/bed-toned, the centreline reads cool/blue.
RIVER_ABSORPTION_RGB = (2.0, 0.9, 0.25)

#: Fraction of light lost to reflection off the water's surface before it
#: ever reaches the bed or the water's own colour -- real water's Fresnel
#: reflectance at a near-vertical viewing angle is only a few percent, but
#: leaving it out would imply every photon makes it through.
RIVER_SURFACE_REFLECTANCE = 0.05


def water_bed_color(depth, bed_albedo=RIVER_BED_ALBEDO,
                     deep_water_color=RIVER_DEEP_WATER_COLOR,
                     absorption_rgb=RIVER_ABSORPTION_RGB,
                     surface_reflectance=RIVER_SURFACE_REFLECTANCE):
    """Depth-dependent base colour for a body of water's bed, as an ``(..., 3)`` array.

    *depth* is the water's absolute depth in metres -- a caller working from
    a ``[0, 1]`` depth fraction (see :attr:`WaterBody.depth`) has to assume a
    physical depth for it first, floored at :data:`RIVER_EDGE_DEPTH_M` (the
    edge, fraction 0) and rising to :data:`RIVER_CENTERLINE_DEPTH_M` (the
    centreline, fraction 1) -- a bank doesn't shelve all the way down to
    nothing. Per channel, light is modelled as travelling down to the bed and back up
    again, attenuated exponentially (Beer-Lambert) by *absorption_rgb*:
    ``T = exp(-k * depth)``. The bed's albedo shows through at fraction ``T``
    and the water's own deep colour makes up the rest, so shallow water
    (``depth`` near 0, ``T`` near 1) reads close to *bed_albedo* and deep
    water (``T`` near 0) reads close to *deep_water_color*. A further
    *surface_reflectance* fraction of light never makes it past the surface
    at all, so the whole mix is scaled down by ``1 - surface_reflectance``.
    Well-defined (not NaN) for every input, including 0.

    This is a pure function of *depth* alone -- it doesn't know or care
    whether a given cell is actually inside a particular water body's mask,
    or whether that body is a river, pond, or anything else; callers (see
    :func:`create_river`) are responsible for restricting where the result
    gets applied.
    """
    depth = np.asarray(depth, dtype='float32')
    absorption_rgb = np.asarray(absorption_rgb, dtype='float32')
    bed_albedo = np.asarray(bed_albedo, dtype='float32')
    deep_water_color = np.asarray(deep_water_color, dtype='float32')
    transmission = np.exp(-absorption_rgb * depth[..., np.newaxis])
    color = bed_albedo * transmission * (1.0 - surface_reflectance) + deep_water_color * (1.0 - transmission)
    return color


class WaterBody:
    """A baked water shape and flow field, with enough to draw itself.

    *mask* is the water's full extent as a boolean ``(rows, cols)`` array --
    including any cell that something drawn afterwards may cover (a bridge
    over a river, say), since water still flows beneath such cover and
    WaterAnimation needs an unbroken source graph across it. *flow_dir* is a
    matching ``(rows, cols, 2)`` array of unit ``(dx, dy)`` vectors, one per
    water cell (undefined elsewhere), giving the direction downstream.
    *blocktype_id* is what :meth:`paint` stamps down. *path*, if the body was
    built from one (see :func:`create_river`), is kept so a level can reach
    its geometry later -- e.g. to find where a road crosses it.

    *depth* is a matching ``(rows, cols)`` float array in ``[0, 1]``: 0 at the
    water's edge, 1 at its centreline, elsewhere (outside the mask) 0. It's a
    pseudo cross-section depth, not a physical measurement -- consumed by a
    depth-dependent colour model (bed albedo showing through in the shallows,
    water's own scattered colour dominating in the deep) and left at zero
    where there's no water. *width* and *half_width* are the per-position
    band width and half-width :func:`create_river` painted (only set when
    built from one -- ``None`` otherwise), one entry per index of *path*'s
    own ``centerline`` (index ``i`` <-> ``path.lo + i`` along ``path.axis``,
    exactly ``path.centerline``'s own indexing): the local edges of the water
    at that position are ``centerline[i] - half_width[i]`` (inclusive) through
    ``centerline[i] - half_width[i] + width[i]`` (exclusive) -- the same band
    the mask was painted with. Kept around so later terrain (sandy banks,
    greenery) can find the water's edge at a given position without
    rescanning the mask.

    A ``create_*`` function (:func:`create_river` and, eventually, siblings
    like a ``create_pond``) is responsible for building this shape/flow data
    -- each free to describe its own kind of water differently -- while
    WaterBody itself stays generic: paint the mask onto a maze, and attach
    the :class:`WaterAnimation` that brings it to life.
    """
    def __init__(self, mask, flow_dir, blocktype_id, path=None, depth=None,
                 width=None, half_width=None):
        self.mask = mask
        self.flow_dir = flow_dir
        self.blocktype_id = blocktype_id
        self.path = path
        self.depth = depth if depth is not None else np.zeros(mask.shape, dtype='float32')
        self.width = width
        self.half_width = half_width
        self.animation = None

    def paint(self, maze):
        """Stamp *blocktype_id* onto *maze* everywhere the mask is set."""
        maze.blocks[self.mask] = self.blocktype_id

    def animate(self, maze, seed=None, start=True):
        """Build and attach the :class:`WaterAnimation` for this body."""
        self.animation = WaterAnimation(maze, self, seed=seed, start=start)
        return self.animation


#: How far the river's width wanders from its nominal value, as a fraction of
#: that nominal width -- see the `widths` meander() call in create_river
#: below. The plan's "meander 10-20% around nominal" is a typical amplitude
#: of wander, not a hard clamp, so individual cells can occasionally stray a
#: bit further; only the hard floor (MIN_RIVER_WIDTH) is actually enforced.
RIVER_WIDTH_AMPLITUDE_FRACTION = 0.15

#: The river must never pinch shut; this is the narrowest it's allowed to get.
MIN_RIVER_WIDTH = 2

#: The cross-section depth profile (0 at the bank, 1 at the centreline) is
#: raised to this power so it reads as a rounded channel bed rather than a
#: V-shaped trench -- flattening the deep middle and steepening the shallows
#: near the banks.
DEPTH_PROFILE_POWER = 1.5

#: Assumed physical depth (metres) of the river at its centreline
#: (WaterBody.depth == 1) -- these rivers don't come with a real depth
#: measurement, so this is what turns that [0, 1] depth fraction into the
#: absolute depth water_bed_color wants.
RIVER_CENTERLINE_DEPTH_M = 4.0

#: Assumed physical depth (metres) at the river's edge (WaterBody.depth == 0)
#: -- a real riverbank doesn't shelve all the way down to nothing, so the
#: edge is floored at this depth rather than mapping to 0 m.
RIVER_EDGE_DEPTH_M = 1.0


def create_river(maze, blocktype_id, rng, start, end, amplitude, wavelength=130, width=4,
                  bounds=None, seed=None, start_animation=True, animate=True, 
                  bed_albedo=RIVER_BED_ALBEDO):
    """Paint a river meandering from *start* to *end* (each an ``(x, y)``
    pair -- typically sharing one coordinate, e.g. a fixed x running the
    height of the room), and return its :class:`WaterBody`.

    The river's centreline is a meandering :class:`~.path.Path` (see
    :func:`~.path.create_path`, which also explains *amplitude*,
    *wavelength* and *bounds*) -- the same curve generator a level's dirt
    paths use, so a river reads as the same kind of terrain feature, just
    with flow added: a unit vector per water cell, tangent to the
    centreline, oriented from *start* toward *end*.

    The river's *width* isn't held constant along its length either: it
    wanders around the nominal *width* argument via its own, shorter-period
    :func:`~.meander.meander` run (so it fluctuates more locally than the
    centreline bends), clamped to :data:`MIN_RIVER_WIDTH` so the channel
    never pinches shut. Alongside the mask, a per-cell depth fraction (0 at
    the bank, 1 at the centreline -- see :data:`DEPTH_PROFILE_POWER`) is
    rasterized into the returned body's ``depth`` (for a later depth-
    dependent colour model), and the per-position width/half-width used to
    paint each cross-section are kept as ``width``/``half_width`` (for later
    terrain -- banks, greenery -- that needs to find the water's edge at a
    given position). See :class:`WaterBody` for exactly how those are
    indexed and used.

    *animate* attaches a :class:`WaterAnimation` (via *seed*/*start_animation*,
    see :meth:`WaterBody.animate`) before returning, which is what most
    callers want. Pass ``animate=False`` when something else still needs to
    draw over part of the river -- a bridge, say -- since the animation's
    display mask is snapshotted from *maze* at attach time and must see the
    map in its final state; call :meth:`WaterBody.animate` explicitly once
    that's done.
    """
    path = create_path(rng, start, end, amplitude, wavelength, width, blocktype_id, bounds=bounds)
    rows, cols = maze.blocks.shape
    length = len(path.centerline)

    # a shorter wavelength than the centerline's own, so the channel's width
    # ripples locally rather than swelling and narrowing over the same long
    # bends the centerline itself takes.
    width_wavelength = max(8, wavelength // 4)
    widths = meander(rng, length, width, width,
                      amplitude=width * RIVER_WIDTH_AMPLITUDE_FRACTION,
                      wavelength=width_wavelength)
    widths = np.maximum(widths, MIN_RIVER_WIDTH)
    half_widths = widths // 2

    # create_path always lays the centerline out from lo to hi along its
    # axis, regardless of which of start/end was larger there -- so the
    # tangent computed below always points in +axis and has to be flipped
    # whenever the caller's start is actually the hi end (i.e. flow runs from
    # hi down to lo).
    start_coord = start[1] if path.axis == 'y' else start[0]
    end_coord = end[1] if path.axis == 'y' else end[0]
    downstream = 1.0 if end_coord >= start_coord else -1.0

    mask = np.zeros((rows, cols), dtype=bool)
    flow_dir = np.zeros((rows, cols, 2), dtype='float32')
    depth = np.zeros((rows, cols), dtype='float32')
    d_perp = np.gradient(path.centerline.astype('float32'))
    for i, c in enumerate(path.centerline):
        coord = path.lo + i
        d = (np.array([d_perp[i], 1.0], dtype='float32') if path.axis == 'y'
             else np.array([1.0, d_perp[i]], dtype='float32'))
        d *= downstream
        d /= np.linalg.norm(d)

        w = int(widths[i])
        half = int(half_widths[i])
        # cross-section depth profile across this cell's band: 1 at the
        # centreline (offset 0), falling to ~0 at the band's outer edge.
        offset = np.arange(w) - half
        cell_depth = np.clip(1.0 - np.abs(offset) / (w / 2.0), 0.0, 1.0) ** DEPTH_PROFILE_POWER

        lo0 = c - half
        hi0 = lo0 + w
        # clip the painted band to the maze's own bounds -- width varying
        # near a room edge can otherwise push a cell negative, which numpy
        # would silently (and wrongly) wrap to the far side of the array
        # rather than raise.
        if path.axis == 'y':
            lo_clip, hi_clip = max(lo0, 0), min(hi0, cols)
            if hi_clip > lo_clip:
                mask[coord, lo_clip:hi_clip] = True
                flow_dir[coord, lo_clip:hi_clip] = d
                depth[coord, lo_clip:hi_clip] = cell_depth[lo_clip - lo0:hi_clip - lo0]
        else:
            lo_clip, hi_clip = max(lo0, 0), min(hi0, rows)
            if hi_clip > lo_clip:
                mask[lo_clip:hi_clip, coord] = True
                flow_dir[lo_clip:hi_clip, coord] = d
                depth[lo_clip:hi_clip, coord] = cell_depth[lo_clip - lo0:hi_clip - lo0]

    body = WaterBody(mask, flow_dir, blocktype_id, path=path, depth=depth,
                      width=widths, half_width=half_widths)
    body.paint(maze)
    # depth-dependent base colour, applied before any animation attaches so
    # WaterAnimation's snapshot of maze.bg_color (its shimmer's base) picks up
    # the gradient rather than the flat blocktype colour -- see water_bed_color.
    # blocktype_id=blocktype_id: mask is the river's full painted extent, but
    # something drawn over it later (a bridge, say) must stop showing this
    # wash once that cell no longer holds this blocktype -- see
    # Maze.wash_bg_color.
    physical_depth = RIVER_EDGE_DEPTH_M + depth * (RIVER_CENTERLINE_DEPTH_M - RIVER_EDGE_DEPTH_M)
    maze.wash_bg_color(water_bed_color(physical_depth, bed_albedo=bed_albedo), amount=1.0,
                        mask=mask, blocktype_id=blocktype_id)
    if animate:
        body.animate(maze, seed=seed, start=start_animation)
    return body


#: Default bank width bound passed to natural_extent() below -- the plan's
#: "sandy banks 0-1 blocks wide". The greenery pass just past it (see
#: GREENERY_MAX_EXTENT) uses a larger bound of its own.
BANK_MAX_EXTENT = 1


class RiverBanks:
    """How much sand :func:`paint_river_banks` actually placed along each
    side of a river, one entry per index of the river's ``path.centerline``
    (the same indexing as :attr:`WaterBody.width`/``half_width`` -- index
    *i* <-> ``path.lo + i``).

    *left* and *right* are per-position counts of sand cells placed
    immediately outside the water's edge on that side, along the path's
    perpendicular axis: *left* just below (lower-coordinate side of)
    ``centerline[i] - half_width[i]`` (the water's near edge), *right* just
    at and above ``centerline[i] - half_width[i] + width[i]`` (its far
    edge) -- see :class:`WaterBody` for that indexing.

    Each count is how many cells were *actually* stamped -- clipped by the
    grass-only guard and the maze's own bounds, so it can be less than what
    :func:`~.meander.natural_extent` originally asked for. A later terrain
    pass (greenery) can start its own band exactly ``left[i]``
    (respectively ``right[i]``) cells past the water's edge on that side,
    without re-deriving any of this.
    """
    def __init__(self, left, right):
        self.left = left
        self.right = right


def paint_river_banks(maze, bt, water_body, rng, max_extent=BANK_MAX_EXTENT):
    """Stamp sandy banks along both sides of *water_body*'s river.

    *water_body* must carry a ``path`` and ``width``/``half_width`` (i.e.
    built by :func:`create_river`). At every position *i* along the path
    (``coord = path.lo + i``), :func:`~.meander.natural_extent` decides --
    independently for each side, via two separate calls sharing *rng* --
    how many cells of sand (0..*max_extent*) to place immediately outside
    that side's edge of the water, so the band drifts smoothly along the
    river's length rather than flickering cell to cell.

    A cell only actually becomes sand if it is currently ``grass`` (a
    building, path, bridge, or the river itself is never overwritten --
    the same idiom as :func:`~.buildings.place_building`'s "only touch
    clear grass") and lies within the maze's own bounds (clipped
    independently of the river's own painted extent, the same way
    :func:`create_river` clips its own cross-sections -- a bank near a room
    edge could otherwise run past it).

    Returns a :class:`RiverBanks` recording how many cells were actually
    placed on each side at each position, for a later terrain pass
    (greenery) to build on without rescanning the mask.
    """
    path = water_body.path
    length = len(path.centerline)
    left_wanted = natural_extent(rng, length, max_extent)
    right_wanted = natural_extent(rng, length, max_extent)

    blocks = maze.blocks
    rows, cols = blocks.shape
    perp_size = cols if path.axis == 'y' else rows

    grass_id = bt.id_of('grass')
    sand_id = bt.id_of('sand')
    left_placed = np.zeros(length, dtype=int)
    right_placed = np.zeros(length, dtype=int)

    for i, c in enumerate(path.centerline):
        coord = path.lo + i
        half = int(water_body.half_width[i])
        w = int(water_body.width[i])
        near_edge = int(c) - half   # inclusive low-coordinate edge of the water
        far_edge = near_edge + w    # exclusive high-coordinate edge of the water

        left_lo, left_hi = near_edge - int(left_wanted[i]), near_edge
        right_lo, right_hi = far_edge, far_edge + int(right_wanted[i])

        # clip each candidate band to the maze's own bounds on the
        # perpendicular axis -- see create_river's identical concern.
        left_lo, left_hi = max(left_lo, 0), min(left_hi, perp_size)
        right_lo, right_hi = max(right_lo, 0), min(right_hi, perp_size)

        if path.axis == 'y':
            band = lambda lo, hi: blocks[coord, lo:hi]
        else:
            band = lambda lo, hi: blocks[lo:hi, coord]

        if left_hi > left_lo:
            cells = band(left_lo, left_hi)
            grass_here = cells == grass_id
            cells[grass_here] = sand_id
            left_placed[i] = grass_here.sum()
        if right_hi > right_lo:
            cells = band(right_lo, right_hi)
            grass_here = cells == grass_id
            cells[grass_here] = sand_id
            right_placed[i] = grass_here.sum()

    return RiverBanks(left_placed, right_placed)


#: The plan's "lush greenery 0-3 blocks wide" -- wider than a bank, since it's
#: meant to read as a soft, drifting fringe rather than a hard 0-1 cell edge.
GREENERY_MAX_EXTENT = 3

#: A deep, saturated green for the lush growth right past a riverbank --
#: distinctly richer than both plain grass (blocktypes.py's grass bg_color,
#: ~(0.01, 0.16, 0.02)) and GRASS_WASH_RAMP_RGB's own "deep green" parched-
#: patch stop (~(0.02, 0.18, 0.03), itself close to plain grass): a
#: noticeably brighter, more saturated green so riverside lushness reads as
#: its own effect rather than another parched-grass variant.
GREENERY_COLOR = (0.02, 0.32, 0.06)

#: Wash strength at the cell immediately past the bank -- deliberately much
#: stronger than grass.GRASS_WASH_AMOUNT (0.15), since this is meant to read
#: as a pronounced riverside lushness effect, not a subtle tint. The colour
#: field itself (not this amount) is what tapers across the band -- see
#: paint_river_greenery -- so this amount applies uniformly while the field
#: blended in fades from GREENERY_COLOR to plain grass.
GREENERY_WASH_AMOUNT = 0.6


class RiverGreenery:
    """How wide a lush-greenery band :func:`paint_river_greenery` actually
    painted along each side of a river, one entry per index of the river's
    ``path.centerline`` -- the same indexing as :class:`RiverBanks`.

    *left* and *right* are per-position counts of grass cells whose colour
    was actually washed, starting immediately past that side's actually-
    placed bank (see :attr:`RiverBanks.left`/``.right``) and running
    outward. Each count is clipped the same way :class:`RiverBanks`' own
    counts are -- by the grass-only guard and the maze's own bounds -- so it
    can be less than what :func:`~.meander.natural_extent` asked for.
    """
    def __init__(self, left, right):
        self.left = left
        self.right = right


def paint_river_greenery(maze, bt, water_body, banks, rng, max_extent=GREENERY_MAX_EXTENT):
    """Wash a lush green band into the grass just past *water_body*'s banks.

    *water_body* must carry a ``path`` and ``width``/``half_width`` (i.e.
    built by :func:`create_river`), and *banks* is the
    :class:`RiverBanks` :func:`paint_river_banks` returned for it -- this is
    the direct sequel to that pass, so each side's band starts exactly
    ``banks.left[i]`` (respectively ``banks.right[i]``) cells past the
    water's edge, i.e. right where the actually-placed sand ends, never
    overlapping it or jumping a gap if a bank got clipped short.

    At every position *i* along the path, :func:`~.meander.natural_extent`
    decides -- independently for each side, via two separate calls sharing
    *rng* -- how many cells (0..*max_extent*) the band wants to reach, the
    same idiom :func:`paint_river_banks` uses for its own width. Only cells
    that are currently ``grass`` are touched (a building, path, sand or the
    river itself is never overwritten), clipped to the maze's own bounds.

    Unlike a bank, greenery doesn't change the blocktype -- grass stays
    grass -- it only blends :data:`GREENERY_COLOR` into the background
    colour via a single :meth:`~..maze.Maze.wash_bg_color` call, at a fixed
    :data:`GREENERY_WASH_AMOUNT`. The *fade* from strong (right past the
    bank) to no effect (at the band's outer edge) is instead baked directly
    into the colour field passed to that call: each cell's colour is
    :data:`GREENERY_COLOR` linearly blended toward grass's own base colour,
    by how far it sits from the bank edge relative to *max_extent* -- so the
    outermost possible cell ends up blended fully back to grass's colour,
    at which point washing it in at any amount has no visible effect.

    Returns a :class:`RiverGreenery` recording how many cells were actually
    washed on each side at each position.
    """
    path = water_body.path
    length = len(path.centerline)
    left_wanted = natural_extent(rng, length, max_extent)
    right_wanted = natural_extent(rng, length, max_extent)

    blocks = maze.blocks
    rows, cols = blocks.shape
    perp_size = cols if path.axis == 'y' else rows

    grass_id = bt.id_of('grass')
    grass_color = np.asarray(bt.get('grass')['bg_color'][:3], dtype='float32')
    lush_color = np.asarray(GREENERY_COLOR, dtype='float32')
    # normalizes the taper against the band's full possible reach (rather
    # than however far any one position's band actually got placed), so the
    # colour field genuinely bottoms out at grass's own colour exactly at
    # the outermost cell a band could ever occupy.
    taper_span = max(max_extent - 1, 1)

    field = np.zeros(maze.shape + (3,), dtype='float32')
    mask = np.zeros(maze.shape, dtype=bool)
    left_placed = np.zeros(length, dtype=int)
    right_placed = np.zeros(length, dtype=int)

    for i, c in enumerate(path.centerline):
        coord = path.lo + i
        half = int(water_body.half_width[i])
        w = int(water_body.width[i])
        near_edge = int(c) - half   # water's inclusive low-coordinate edge
        far_edge = near_edge + w    # water's exclusive high-coordinate edge

        # each side's band starts exactly where that side's actually-placed
        # bank ends, not at the water's edge itself.
        left_start = near_edge - int(banks.left[i])
        right_start = far_edge + int(banks.right[i])

        left_lo, left_hi = left_start - int(left_wanted[i]), left_start
        right_lo, right_hi = right_start, right_start + int(right_wanted[i])

        # clip each candidate band to the maze's own bounds on the
        # perpendicular axis -- see create_river's identical concern.
        left_lo, left_hi = max(left_lo, 0), min(left_hi, perp_size)
        right_lo, right_hi = max(right_lo, 0), min(right_hi, perp_size)

        if path.axis == 'y':
            band = lambda lo, hi: blocks[coord, lo:hi]
        else:
            band = lambda lo, hi: blocks[lo:hi, coord]

        if left_hi > left_lo:
            # offset 0 is the cell nearest the bank (left_hi - 1); it grows
            # toward the band's far, outer end.
            offset = (left_hi - 1) - np.arange(left_lo, left_hi)
            fraction = np.clip(offset / taper_span, 0.0, 1.0)[:, None]
            colors = lush_color * (1 - fraction) + grass_color * fraction

            grass_here = band(left_lo, left_hi) == grass_id
            if path.axis == 'y':
                field[coord, left_lo:left_hi] = colors
                mask[coord, left_lo:left_hi] = grass_here
            else:
                field[left_lo:left_hi, coord] = colors
                mask[left_lo:left_hi, coord] = grass_here
            left_placed[i] = grass_here.sum()

        if right_hi > right_lo:
            # offset 0 is the cell nearest the bank (right_lo); it grows
            # toward the band's far, outer end.
            offset = np.arange(right_lo, right_hi) - right_lo
            fraction = np.clip(offset / taper_span, 0.0, 1.0)[:, None]
            colors = lush_color * (1 - fraction) + grass_color * fraction

            grass_here = band(right_lo, right_hi) == grass_id
            if path.axis == 'y':
                field[coord, right_lo:right_hi] = colors
                mask[coord, right_lo:right_hi] = grass_here
            else:
                field[right_lo:right_hi, coord] = colors
                mask[right_lo:right_hi, coord] = grass_here
            right_placed[i] = grass_here.sum()

    maze.wash_bg_color(field, GREENERY_WASH_AMOUNT, mask=mask)
    return RiverGreenery(left_placed, right_placed)


class WaterAnimation:
    """Animates one WaterBody's surface.

    *maze* is the Maze the water lives on. *water_body* is the
    :class:`WaterBody` whose ``mask`` (the water's full extent, including any
    cell later covered by a bridge -- only the subset still showing
    *water_body.blocktype_id* today is actually painted, see below) and
    ``flow_dir`` (the direction downstream, one unit vector per water cell)
    drive the simulation below.

    From the mask and flow field this builds a sparse "source" matrix: for
    each simulated cell, its few upstream neighbours (also water, weighted by
    how closely each aligns with the cell's own flow direction -- see
    :meth:`_build_source_matrix`). A background thread then repeatedly ages
    the noise field toward a blend of itself, that weighted upstream average,
    and fresh randomness, so texture drifts downstream over time rather than
    just flickering in place. Each step's noise is colour-mapped and pushed
    to the display through a single ColorModifier spanning every *painted*
    cell (bridge cells are simulated but never painted -- the bridge tile
    covers them).
    """
    #: noise(t+1) = SELF_WEIGHT*noise(t) + NEIGHBOR_WEIGHT*neighbor_noise(t) + RANDOM_WEIGHT*random
    SELF_WEIGHT = 0.4
    NEIGHBOR_WEIGHT = 0.4
    RANDOM_WEIGHT = 0.2

    #: *random* above is drawn as ``uniform(0, 1) ** RANDOM_SKEW`` rather than
    #: flat: since SELF_WEIGHT + NEIGHBOR_WEIGHT sum to 0.9, the noise field's
    #: steady-state mean converges to E[random] (the 0.9 damping and the 0.1
    #: weight exactly cancel), so this is also, in effect, where the *river's*
    #: steady-state mean gets set on the colour ramp. Skewing it toward 0
    #: pulls that mean toward the ramp's blue end while still leaving an
    #: occasional high draw to flash toward foam.
    RANDOM_SKEW = 2.0

    #: real-time interval between noise updates. Water doesn't need to
    #: shimmer at frame rate; slow enough to read as a lazy river, fast
    #: enough that drifting downstream is visible.
    STEP_INTERVAL = 0.1

    def __init__(self, maze, water_body, seed=None, start=True):
        self.maze = maze
        self.water_body = water_body
        self.water_mask = water_body.mask
        self.flow_dir = water_body.flow_dir
        self._done = False
        self._thread = None
        self._rng = np.random.RandomState(seed)

        rows, cols = self.water_mask.shape
        # the full simulated domain, including cells now hidden under a
        # bridge -- water still flows there, and the graph needs to be
        # unbroken for noise to reach the far bank.
        self._cells = np.flatnonzero(self.water_mask.ravel())
        self.noise = self._skewed_random(len(self._cells))
        self._source_matrix = self._build_source_matrix(rows, cols)

        # only cells still showing as this body's blocktype today (i.e. not
        # overdrawn by a bridge) are ever pushed to the display.
        paint_mask = self.water_mask & (maze.blocks == water_body.blocktype_id)
        paint_cells = np.flatnonzero(paint_mask.ravel())
        # self._cells and paint_cells are both ascending (flatnonzero order),
        # so searchsorted maps each painted cell to its row in self.noise.
        self._paint_index = np.searchsorted(self._cells, paint_cells)

        self._color_modifier = ColorModifier()
        maze.add_area_modifier(paint_cells, self._color_modifier)

        self._restyle()
        if start:
            self._start()

    def _build_source_matrix(self, rows, cols):
        """A sparse ``(N, N)`` row-stochastic matrix over the simulated cells.

        Row i's nonzero entries are cell i's upstream neighbours (limited to
        cells in *water_mask* -- a bank cell never supplies noise), weighted
        by how closely each aligns with cell i's own flow direction, so a
        neighbour straight upstream counts more than one just off to the
        side. A cell with no upstream neighbour in the mask (the river's own
        source end) supplies itself, so the update formula still behaves
        there.
        """
        cells = self._cells
        n = len(cells)
        ys, xs = np.divmod(cells, cols)
        index_of = -np.ones(rows * cols, dtype='int64')
        index_of[cells] = np.arange(n)

        src_list, nbr_list, weight_list = [], [], []
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        for dy, dx in offsets:
            ny, nx = ys + dy, xs + dx
            in_bounds = (ny >= 0) & (ny < rows) & (nx >= 0) & (nx < cols)
            is_water = np.zeros(n, dtype=bool)
            is_water[in_bounds] = self.water_mask[ny[in_bounds], nx[in_bounds]]
            if not is_water.any():
                continue

            src_idx = np.flatnonzero(is_water)
            nbr_idx = index_of[ny[src_idx] * cols + nx[src_idx]]

            # neighbour (dx, dy) away from src is upstream of it when src's
            # own flow points away from that neighbour -- i.e. the neighbour
            # sits roughly opposite the flow direction. alignment is the
            # cosine of that backward angle: positive (and largest) for a
            # neighbour straight upstream, negative for one downstream.
            off = np.array([dx, dy], dtype='float32')
            off /= np.linalg.norm(off)
            fdir = self.flow_dir[ys[src_idx], xs[src_idx]]
            alignment = -(fdir * off).sum(axis=1)

            keep = alignment > 0
            if not keep.any():
                continue
            src_list.append(src_idx[keep])
            nbr_list.append(nbr_idx[keep])
            weight_list.append(alignment[keep])

        if src_list:
            src = np.concatenate(src_list)
            nbr = np.concatenate(nbr_list)
            weight = np.concatenate(weight_list)
        else:
            src = nbr = weight = np.zeros(0, dtype='float64')

        matrix = scipy.sparse.coo_matrix((weight, (src, nbr)), shape=(n, n)).tocsr()
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()

        # a cell with no upstream water neighbour (the source end of the
        # river) supplies itself, so neighbor_noise falls back to its own
        # noise there rather than zero.
        sourceless = np.flatnonzero(row_sum == 0)
        if len(sourceless):
            self_loops = scipy.sparse.coo_matrix(
                (np.ones(len(sourceless)), (sourceless, sourceless)), shape=(n, n))
            matrix = (matrix + self_loops).tocsr()
            row_sum = np.asarray(matrix.sum(axis=1)).ravel()

        return scipy.sparse.diags(1.0 / row_sum) @ matrix

    def _skewed_random(self, size):
        """*size* draws of the update's random term; see RANDOM_SKEW."""
        return (self._rng.uniform(0.0, 1.0, size=size) ** self.RANDOM_SKEW).astype('float32')

    def advance(self):
        """One noise update step. Runs on the animation thread, but exposed
        directly so tests can drive it without sleeping."""
        neighbor_noise = self._source_matrix @ self.noise
        random_term = self._skewed_random(self.noise.shape)
        self.noise = (self.SELF_WEIGHT * self.noise
                      + self.NEIGHBOR_WEIGHT * neighbor_noise
                      + self.RANDOM_WEIGHT * random_term)
        np.clip(self.noise, 0.0, 1.0, out=self.noise)
        self._restyle()

    def _restyle(self):
        """Colour-map the current noise and push it out to the display."""
        t = self.noise[self._paint_index]
        channels = [np.interp(t, WATER_COLOR_RAMP_T, [rgb[i] for rgb in WATER_COLOR_RAMP_RGB])
                    for i in range(3)]
        color = np.stack(channels, axis=-1).astype('float32')

        delta = np.zeros((len(t), 4), dtype='float32')
        delta[:, :3] = color
        self._color_modifier.bgcolor = delta
        if self._color_modifier._slot is not None:
            self._color_modifier._slot._dirty = True
        self.maze.appearance_changed()

    def _start(self):
        self._thread = threading.Thread(target=self._run, name='water-animation', daemon=True)
        self._thread.start()

    def _run(self):
        while not self._done:
            time.sleep(self.STEP_INTERVAL)
            self.advance()

    def destroy(self):
        """Stop animating and detach from the maze. Idempotent."""
        if self._done:
            return
        self._done = True
        self.maze.remove_area_modifier(self._color_modifier)
