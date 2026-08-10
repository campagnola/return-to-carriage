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

    A ``create_*`` function (:func:`create_river` and, eventually, siblings
    like a ``create_pond``) is responsible for building this shape/flow data
    -- each free to describe its own kind of water differently -- while
    WaterBody itself stays generic: paint the mask onto a maze, and attach
    the :class:`WaterAnimation` that brings it to life.
    """
    def __init__(self, mask, flow_dir, blocktype_id, path=None):
        self.mask = mask
        self.flow_dir = flow_dir
        self.blocktype_id = blocktype_id
        self.path = path
        self.animation = None

    def paint(self, maze):
        """Stamp *blocktype_id* onto *maze* everywhere the mask is set."""
        maze.blocks[self.mask] = self.blocktype_id

    def animate(self, maze, seed=None, start=True):
        """Build and attach the :class:`WaterAnimation` for this body."""
        self.animation = WaterAnimation(maze, self, seed=seed, start=start)
        return self.animation


def create_river(maze, blocktype_id, rng, start, end, amplitude, wavelength=130, width=4,
                  bounds=None, seed=None, start_animation=True, animate=True):
    """Paint a river meandering from *start* to *end* (each an ``(x, y)``
    pair -- typically sharing one coordinate, e.g. a fixed x running the
    height of the room), and return its :class:`WaterBody`.

    The river's centreline is a meandering :class:`~.path.Path` (see
    :func:`~.path.create_path`, which also explains *amplitude*,
    *wavelength* and *bounds*) -- the same curve generator a level's dirt
    paths use, so a river reads as the same kind of terrain feature, just
    with flow added: a unit vector per water cell, tangent to the
    centreline, with +axis taken as downstream (an arbitrary but fixed
    choice).

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
    half = path.width // 2

    mask = np.zeros((rows, cols), dtype=bool)
    flow_dir = np.zeros((rows, cols, 2), dtype='float32')
    d_perp = np.gradient(path.centerline.astype('float32'))
    for i, c in enumerate(path.centerline):
        coord = path.lo + i
        d = (np.array([d_perp[i], 1.0], dtype='float32') if path.axis == 'y'
             else np.array([1.0, d_perp[i]], dtype='float32'))
        d /= np.linalg.norm(d)
        if path.axis == 'y':
            y, x0 = coord, c - half
            mask[y, x0:x0 + width] = True
            flow_dir[y, x0:x0 + width] = d
        else:
            x, y0 = coord, c - half
            mask[y0:y0 + width, x] = True
            flow_dir[y0:y0 + width, x] = d

    body = WaterBody(mask, flow_dir, blocktype_id, path=path)
    body.paint(maze)
    if animate:
        body.animate(maze, seed=seed, start=start_animation)
    return body


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
    SELF_WEIGHT = 0.5
    NEIGHBOR_WEIGHT = 0.4
    RANDOM_WEIGHT = 0.1

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
        self._base_bg = maze.bg_color.reshape(-1, 4)[paint_cells].astype('float32')

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
        delta[:, :3] = color - self._base_bg[:, :3]
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
