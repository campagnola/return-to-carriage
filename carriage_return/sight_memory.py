"""How long a glimpsed wall face stays remembered.

Every wall-face texel of a level carries its own decay clock. A memory fades
linearly from the brightness it was last seen at to nothing over its
``decay_time``; after that it has expired. Each time a face comes back into
view its retention grows, and a better-retained memory lasts longer::

    retention  += MEMORY_RETENTION_GAIN * age      (age: elapsed fraction, 0..1)
    decay_time  = MEMORY_BASE_DECAY_TIME ** retention

so a face first seen fresh (age 1, the default expired state) gains the full
gain, and one revisited half way through its decay gains half of it. Glimpses
crammed together barely help; spaced ones build long-term memory, up to
``MEMORY_MAX_RETENTION``.

Decay is timestamp based: a texel records *when* it was last in view, and its
fade is worked out from ``now - seen_at`` whenever the level is drawn. Levels
the player is not on do no work, yet the time spent away still counts once the
player returns.

Game-side module: no rendering library may be imported here.
"""
import numpy as np

#: Decay time (seconds) at retention 1; raised to the retention, so must be > 1.
MEMORY_BASE_DECAY_TIME = 16.0

#: Retention a never-seen (or fully expired) face starts from.
MEMORY_MIN_RETENTION = 1.0

#: Retention a memory can grow to; at the 16 s base this lasts ~24 hours.
MEMORY_MAX_RETENTION = 4.1

#: Display-space brightness (as ``seen``, 0..MEMORY_MAX_BRIGHTNESS) a wall face
#: must show to count as in view. Anything dimmer is out of view: it neither
#: refreshes its memory nor grows its retention, so faint spill or a fading glow cannot hold a face
#: "in view" and swallow the next flash of light as the same sighting.
MEMORY_VISIBLE_THRESHOLD = 0.03

#: Retention gained on coming back into view with the memory fully expired;
#: scaled by the elapsed fraction of the decay time otherwise.
MEMORY_RETENTION_GAIN = 0.4


class SightMemory:
    """Decay state for the wall-face texels of one level's memory field.

    State is kept only for the texels of *face_mask* (memory never lives
    anywhere else), as compact 1-D arrays in ``np.flatnonzero(face_mask)``
    order:

    - ``retention``: the decay exponent, MEMORY_MIN_RETENTION..MEMORY_MAX_RETENTION
    - ``decay_time``: seconds to fade to nothing, ``BASE ** retention``
    - ``seen_at``: clock time last in view (float64; ``-inf`` = never)
    - ``peak``: display-space brightness when last in view
    - ``in_view``: in view on the previous update, for the rising edge
    """

    def __init__(self, face_mask):
        self.faces = np.flatnonzero(face_mask)
        n = len(self.faces)
        self.retention = np.full(n, MEMORY_MIN_RETENTION, dtype='float32')
        self.decay_time = np.full(n, MEMORY_BASE_DECAY_TIME ** MEMORY_MIN_RETENTION,
                                  dtype='float32')
        self.seen_at = np.full(n, -np.inf, dtype='float64')
        self.peak = np.zeros(n, dtype='float32')
        self.in_view = np.zeros(n, dtype=bool)

    def age(self, now):
        """Elapsed fraction of each face's decay time at *now*, clipped to 0..1."""
        return np.clip((now - self.seen_at) / self.decay_time, 0.0, 1.0)

    def lose_sight(self):
        """Nothing is in view any more, so the next sighting is a fresh one."""
        self.in_view[:] = False

    def update(self, now, memory, seen=None):
        """Advance to clock time *now* and write the remembered brightness of
        every face into the ``(h, w)`` field *memory* (in place).

        *seen* is the ``(h, w)`` display-space brightness in view now (already
        capped), or None when nothing is in view. A face counts as in view
        when it shows at least MEMORY_VISIBLE_THRESHOLD: a wall in sight but
        too dim to make out neither refreshes its memory nor grows its retention.
        """
        age = self.age(now)
        remembered = (self.peak * (1.0 - age)).astype('float32')
        if seen is None:
            self.lose_sight()
        else:
            seen = seen.reshape(-1)[self.faces]
            view = seen >= MEMORY_VISIBLE_THRESHOLD
            rising = view & ~self.in_view
            if rising.any():
                retention = np.minimum(MEMORY_MAX_RETENTION,
                                       self.retention[rising] + MEMORY_RETENTION_GAIN * age[rising])
                self.retention[rising] = retention
                self.decay_time[rising] = MEMORY_BASE_DECAY_TIME ** retention
            # a dimmer glimpse restarts the clock but never lowers the memory
            remembered[view] = np.maximum(remembered[view], seen[view])
            self.peak[view] = remembered[view]
            self.seen_at[view] = now
            self.in_view[:] = view
        memory.reshape(-1)[self.faces] = remembered
