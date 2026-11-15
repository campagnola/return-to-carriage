"""Ruined buildings: wall rectangles with broken perimeters, stamped onto
whatever clear ground a level offers them."""
import numpy as np


class Building:
    """Where a placed building stands: its footprint in maze cells, walls
    included, as top-left corner *(x0, y0)* and size *(w, h)*."""

    def __init__(self, x0, y0, w, h):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h

    def contains(self, pos):
        """True if cell *pos* ``(x, y)`` lies within the footprint."""
        x, y = pos
        return self.x0 <= x < self.x0 + self.w and self.y0 <= y < self.y0 + self.h

    def field_slice(self, supersample):
        """The footprint as a ``(rows, cols)`` slice into a field sampled
        *supersample* texels per cell (see :class:`~..world.Level`)."""
        ss = supersample
        return (slice(self.y0 * ss, (self.y0 + self.h) * ss),
                slice(self.x0 * ss, (self.x0 + self.w) * ss))

    def __repr__(self):
        return 'Building(x0=%d, y0=%d, w=%d, h=%d)' % (self.x0, self.y0, self.w, self.h)


def place_building(blocks, bt, rng, cx, cy, x_bounds, y_bounds):
    """Stamp one dilapidated building centred near *(cx, cy)*; return its
    :class:`Building`, or None if it could not be placed.

    A building is a wall rectangle with a dirt-path floor and 2-4 gaps broken
    into its perimeter (each 1-6 cells long), so it reads as a ruin rather than
    an intact structure. Skipped (returns None) if its footprint, plus a
    1-cell buffer, is not all clear grass -- which keeps it off a river, a
    bridge, a path, or another building.
    """
    grass_id = bt.id_of('grass')
    wall_id = bt.id_of('wall')
    dirt_id = bt.id_of('dirt')

    w = rng.randint(7, 13)
    h = rng.randint(6, 10)
    x0 = int(np.clip(cx - w // 2, x_bounds[0], x_bounds[1] - w + 1))
    y0 = int(np.clip(cy - h // 2, y_bounds[0], y_bounds[1] - h + 1))

    px0, px1 = max(x_bounds[0], x0 - 1), min(x_bounds[1], x0 + w)
    py0, py1 = max(y_bounds[0], y0 - 1), min(y_bounds[1], y0 + h)
    if not (blocks[py0:py1 + 1, px0:px1 + 1] == grass_id).all():
        return None

    footprint = blocks[y0:y0 + h, x0:x0 + w]
    footprint[0, :] = wall_id
    footprint[-1, :] = wall_id
    footprint[:, 0] = wall_id
    footprint[:, -1] = wall_id
    footprint[1:-1, 1:-1] = dirt_id

    # Walk the perimeter in order so a gap can run across a corner.
    perimeter = (
        [(y0, x) for x in range(x0, x0 + w)]
        + [(y, x0 + w - 1) for y in range(y0 + 1, y0 + h)]
        + [(y0 + h - 1, x) for x in range(x0 + w - 2, x0 - 1, -1)]
        + [(y, x0) for y in range(y0 + h - 2, y0, -1)]
    )
    for _ in range(rng.randint(2, 5)):
        start = rng.randint(len(perimeter))
        gap_len = rng.randint(1, 7)
        for k in range(gap_len):
            r, c = perimeter[(start + k) % len(perimeter)]
            blocks[r, c] = dirt_id

    return Building(x0, y0, w, h)


def try_place_building(blocks, bt, rng, cx, cy, x_bounds, y_bounds, tries=15):
    """Attempt to place a building near *(cx, cy)*, jittering the centre on
    each retry so a spot crowded by a path, a river, or another building
    still usually finds room nearby. Returns the placed :class:`Building`,
    or None if every try failed."""
    for attempt in range(tries):
        jx = cx if attempt == 0 else cx + rng.randint(-5, 6)
        jy = cy if attempt == 0 else cy + rng.randint(-5, 6)
        building = place_building(blocks, bt, rng, jx, jy, x_bounds, y_bounds)
        if building is not None:
            return building
    return None
