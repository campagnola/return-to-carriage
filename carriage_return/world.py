"""The world: several levels, and the portals that join them.

A :class:`Level` is one maze plus a name. A :class:`LevelPortal` joins two
levels through a :class:`~.portal.PortalEnd` on each side. A portal end is an
entity that lives on the map like any item (see :mod:`.portal`); it is what you
see and step onto, and it carries how it is used (walk-on vs command) and
whether you may leave through it. The world only has to hold the *join* -- the
pair of ends and which levels they connect.

Positions here are ``(x, y)`` -- the same convention as entity location slots
and the opposite of numpy's ``maze.blocks[y, x]`` indexing.

Game-side module: no rendering library may be imported here.
"""
import numpy as np

from .array_cache import ArraySumCache
from .blocktypes import BlockTypes
from .events import Observable
from .layers import FieldLayer


#: Resolution of the sight fields relative to maze cells. One number, shared by
#: Level (which allocates the fields) and Scene (which composites them), so the
#: two cannot disagree about how big a level's fields are.
SIGHT_SUPERSAMPLE = 4

#: Rec. 709 luminance weights. Used to collapse an RGB value to a single
#: perceived brightness -- both for the eye-adaptation target and for the
#: per-cell material reflectance the albedo map holds.
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype='float32')

#: Reference luminance and strength for the memory overlay. Memory is mapped to
#: a display value through a *fixed* exposure (``0.18 / MEMORY_REF_LUMINANCE``),
#: deliberately independent of the player's live adaptation: a remembered area
#: is a faint recollection, not something that should brighten just because the
#: eye is now dark-adapted. MEMORY_STRENGTH caps how bright the recollection
#: gets on screen. In reflected-luminance units (cd/m^2), on the same scale as
#: the eye's adaptation; tunable in the visual pass.
MEMORY_REF_LUMINANCE = 0.64
MEMORY_STRENGTH = 1.0


class Level:
    """One maze, under a name, plus everything sized against that maze.

    A Level owns *every* array sized to its maze -- the line-of-sight and
    memory fields, the composited lighting, the ``light`` and ``memory_overlay``
    fields the renderer uploads, and (injected by the display backend) the
    ``visibility`` shadow provider. Nothing maze-sized lives on the scene "for
    whichever level is current"; it all lives here, on the level it belongs to.

    That is what makes the level the unit of consistency across threads. A
    reader -- the draw thread compositing a frame while the input thread walks
    the player to another level -- captures one Level reference and derives
    every shape from it. ``level.line_of_sight``, ``level.illuminance`` and
    ``level.visibility`` are all sized to ``level.maze`` by construction, so no
    interleaving can hand back a field and a maze that disagree. The worst a
    torn read costs is a frame drawn from a stale-but-internally-consistent
    level; :meth:`update_sight` turns even that into a fully-blocked (memory
    only) frame, because the player is not on the level being drawn.

    ``memory`` is per level for the same reason it is useful: what you saw of
    a level is a fact about that level, and survives going elsewhere and
    coming back.
    """

    #: sight memory fades to this fraction of itself per second (equivalent to
    #: the historical 0.999-per-frame decay at 60 fps)
    MEMORY_DECAY_RATE = 0.999 ** 60

    def __init__(self, name, maze, supersample=SIGHT_SUPERSAMPLE):
        self.name = name
        self.maze = maze
        self.world = None
        self.supersample = supersample

        # Named cells of interest on this level, ``{name: (x, y)}`` -- the start
        # square, portal mouths, torch stands. A higher-numbered level reads the
        # entries of the lower-numbered levels it links back to (see the level
        # builders in :mod:`.levels`), so the coordinates portals hang from live
        # with the level that owns them, not in the code that wires them.
        self.locations = {}

        # let anything holding a maze find the level it belongs to; this is
        # the hop that lets an entity ask about *its own* level's sight
        maze.level = self

        ms = maze.shape
        h, w = ms[0] * supersample, ms[1] * supersample
        # Line of sight and lighting are still three-channel (an RGB shadow map
        # times RGB light); memory is a single linear luminance per cell.
        self.field_shape = (h, w, 3)
        self.memory = np.zeros((h, w), dtype='float32')
        self.line_of_sight = np.zeros(self.field_shape, dtype='float32')

        # Light components whose global location is on this level; each Light
        # adds and removes itself as its host moves (see Light._register).
        self.lights = []

        # Fired when a light on this level changes what it emits. The display
        # backend connects this to its repaint (via Scene.request_redraw): the
        # level owns the decision to recomposite, the scene owns the repaint,
        # and neither the light nor the level needs a scene reference to reach
        # the other. See add_light/_light_changed.
        self.lighting_changed = Observable()

        # Composited HDR illuminance for this level (the linear sum of its light
        # maps, RGB), and its cross-frame cache. This is the expensive step; it is rebuilt only
        # when marked dirty (a lighting or viewpoint change), never per frame.
        self.illuminance = None
        self._illuminance_dirty = False
        self.light_cache = ArraySumCache()

        # Per-cell material reflectance luminance at field resolution, (h, w, 1).
        # Built once from the fixed maze -- each block id maps to the luminance
        # of its base ``bg_color`` -- and kept, since the maze never changes. It
        # relates arriving light (illuminance) to reflected luminance for the
        # eye-adaptation target and the memory field.
        self._albedo_lum = None

        # The two composited fields the renderer uploads, each owned by the
        # level so its identity is stable for a backend that captured it and
        # always the right shape for this maze:
        #  - light: RGBA float32. Channels [0:3] are the raw linear HDR
        #    illuminance E (NOT gated by line of sight); channel [3] is the
        #    line-of-sight scalar in 0..1. The GPU multiplies the two, so
        #    reflection and emission are both gated by line of sight there.
        #  - memory_overlay: single-channel float32, the display-space memory
        #    overlay ``mem_disp * (1 - los)``, already gamma-encoded and masked
        #    to the cells not currently in view.
        self.light = FieldLayer('light', shape=(h, w, 4))
        self.memory_overlay = FieldLayer('memory', shape=(h, w))

        # Shadow-map provider sized to this maze, injected by the display
        # backend when it builds this level's GL resources (see the vispy
        # renderer's _rebuild_for_level). Duck-typed render(pos, read=True) ->
        # (h, w, >=3) array; no rendering library is imported here to hold it.
        self.visibility = None

        # recompute line of sight on the next update (the viewer just arrived
        # or moved); set true so the first frame casts sight from scratch
        self._need_los_update = True

        # Eye-adaptation bounds this level imposes on the viewer, in reflected
        # luminance (cd/m^2), or None to use the module defaults. update_sight
        # pushes these onto the player's eye each frame the player is here, so a
        # level that says nothing resets the eye to the default range (see
        # EyeAdaptation.set_bounds). Setting both to the *same* value pins the
        # eye -- a fixed exposure that ignores the sampled scene -- which home
        # does, being lit by the sky rather than the dim floor the window sees.
        self.min_adapt_luminance = None
        self.max_adapt_luminance = None

    def clear_line_of_sight(self):
        """Nothing on this level is in sight; the viewer has gone elsewhere.

        Written in place, so the array a concurrent reader holds stays the
        right shape throughout. Called when the level stops being displayed:
        with no player here, no torch on this level is being watched, which is
        what stops the flicker thread burning flames nobody can see.
        """
        self.line_of_sight[:] = 0

    def add_light(self, light):
        """Register *light* as shining on this level.

        Called by :meth:`Light._register` when a light's host moves onto this
        level, or when a map light is pinned here. Besides holding the light in
        ``lights`` for compositing, the level subscribes to the light's
        ``changed`` signal so that a change in the light's colour or brightness
        becomes stale lighting and a repaint here -- which is what lets a light
        announce it changed without holding any reference to the scene.
        """
        self.lights.append(light)
        light.changed.connect(self._light_changed)

    def remove_light(self, light):
        """Take *light* off this level; its host has moved elsewhere."""
        light.changed.disconnect(self._light_changed)
        self.lights.remove(light)

    def _light_changed(self):
        """A light on this level changed what it emits: recomposite and repaint.

        Runs on whichever thread set the light -- notably the torch flicker
        thread -- so it only nulls a reference (``invalidate_lighting``) and
        fires an observable, both safe off the main thread, exactly as the old
        direct calls to invalidate_lighting()/request_redraw() were.
        """
        self.invalidate_lighting()
        self.lighting_changed()

    def invalidate_lighting(self):
        """Mark the composited illuminance dirty; it is rebuilt on the next request.

        Called by a light on this level whose emitted light changed (colour,
        brightness). Nothing needs to be *kept* to make a flickering flame show:
        the field is linear HDR and the GPU exposure varies slowly, so a rebuilt
        ``illuminance`` carrying the flame's new brightness modulates the output
        directly rather than being renormalised away. The previous field is left
        in place (only flagged), so a reader between frames still sees valid
        lighting until illuminance_map recomposites it.
        """
        self._illuminance_dirty = True

    def invalidate_sight(self):
        """The viewer moved on this level: recast sight and recomposite light.

        Flags the composited illuminance dirty and forces line of sight to be
        recast, so the next update rebuilds both for the new viewpoint. The
        illuminance field itself is left in place (see invalidate_lighting): it
        is position-independent, so the stale map still answers "how much light
        reaches this cell" correctly until it is recomposited.
        """
        self._need_los_update = True
        self._illuminance_dirty = True

    def enter(self):
        """Prepare this level to be shown, dropping every cross-frame cache.

        Called when the level becomes the displayed one. Blanks the composited
        fields and the illuminance cache so the first frame is built from
        scratch, matching what a freshly-entered level should look like. The
        albedo map is *not* dropped: it depends only on the fixed maze.
        """
        self.illuminance = None
        self._illuminance_dirty = False
        self._need_los_update = True
        self.light.set_data(np.zeros((*self.memory.shape, 4), dtype='float32'))
        self.memory_overlay.set_data(np.zeros(self.memory.shape, dtype='float32'))

    def _build_albedo_lum(self):
        """Per-cell reflectance luminance at field resolution, ``(h, w, 1)``.

        Built from the *base* blocktype table (``blocktypes.data['bg_color']``),
        not the jittered ``maze.bg_color()``, so it is static and cacheable:
        each block id maps to the luminance of its background colour, the maze's
        block grid indexes that lookup, and the result is nearest-neighbour
        upsampled by ``supersample`` -- the same maze->field scaling an
        ArrayLight uses (see :meth:`ArrayLight._render_light_map`).
        """
        bg = self.maze.blocktypes.data['bg_color'][:, :3]     # (n_blocktypes, 3)
        bt_lum = (bg @ LUMINANCE_WEIGHTS).astype('float32')   # (n_blocktypes,)
        cell_lum = bt_lum[self.maze.blocks]                   # (maze_h, maze_w)
        ss = self.supersample
        up = np.repeat(np.repeat(cell_lum, ss, axis=0), ss, axis=1)
        return up[:, :, None]

    def update_sight(self, dt, player):
        """Advance this level's sight/memory fields by *dt* seconds, writing the
        result into ``self.light`` and ``self.memory_overlay``.

        Called once per rendered frame by the display backend, for the level it
        is currently showing. Everything read here -- the line-of-sight and
        lighting fields, the lights that feed them, the shadow provider -- is
        this one level's, sized to this one maze, so no interleaving with a
        level switch on another thread can compose arrays of two shapes.

        The CPU no longer tone-maps, and it no longer conflates lighting with
        line of sight. The ``light`` field carries raw linear HDR illuminance in
        channels [0:3] (NOT gated by line of sight) and the line-of-sight scalar
        in channel [3]; the GPU gates both reflection and emission by that
        scalar, applies albedo, and runs the Reinhard curve + display gamma
        under the player's eye-adaptation exposure. Keeping line of sight
        separate is what lets a self-emitting glyph in an unlit but in-view cell
        still show -- its emission is gated by line of sight, not by local
        light. This method also drives that adaptation, from the reflected
        luminance of the blocks in a window around the player, and maintains the
        display-space memory overlay in ``memory_overlay``.

        When *player* is not standing on this level the view is fully blocked:
        line of sight is zero, so reflection and emission are gated off on the
        GPU and only the memory overlay survives. That is what the renderer
        shows in the brief window after it has switched to a new level but
        before the player has been moved onto it -- the level's memory, for
        free.
        """
        watched = player is not None and player.level is self
        h, w = self.memory.shape

        if watched:
            if self._need_los_update:
                self.line_of_sight = player.line_of_sight().astype('float32', copy=False)
                self._need_los_update = False
            line_of_sight = self.line_of_sight

            # Composite this level's HDR illuminance, recompositing if a move or
            # a light change marked it dirty. Held in a local because an
            # animator/flicker thread may mark it dirty mid-frame; the worst that
            # costs is one stale frame.
            illuminance = self.illuminance_map()
            if self._albedo_lum is None:
                self._albedo_lum = self._build_albedo_lum()

            # Line of sight is effectively a scalar (opaque occluders, so the
            # shadow map's three channels are identical); collapse it to one.
            los_scalar = line_of_sight.max(axis=2)

            # Reflected luminance per cell (cd/m^2): what the eye and memory
            # respond to. illuminance is lux; a Lambertian surface of reflectance
            # rho lit by E lux has luminance rho*E/pi, so the 1/pi turns arriving
            # light into light leaving the surface toward the eye.
            lumE = illuminance @ LUMINANCE_WEIGHTS
            Y_refl = self._albedo_lum[:, :, 0] * lumE / np.pi

            # This level decides how far the eye may open up or stop down while
            # the player is on it; a level that specifies nothing resets the eye
            # to the default range, and a level that pins both bounds (home)
            # holds the exposure fixed regardless of what the window samples.
            player.adaptation.set_bounds(self.min_adapt_luminance,
                                         self.max_adapt_luminance)

            # Drive eye adaptation from the line-of-sight-weighted mean reflected
            # luminance in a +/-5 maze-cell window around the player. Nothing
            # visible (all shadow) -> keep the previous adaptation.
            x, y = player.location.global_location.slot
            ss = self.supersample
            y0, y1 = max(0, y * ss - 5 * ss), min(h, y * ss + 5 * ss)
            x0, x1 = max(0, x * ss - 5 * ss), min(w, x * ss + 5 * ss)
            win_w = los_scalar[y0:y1, x0:x1]
            wsum = win_w.sum()
            if wsum > 0:
                L_scene = float((Y_refl[y0:y1, x0:x1] * win_w).sum() / wsum)
                player.adaptation.adapt(L_scene, dt)

            # remember the brightest reflected luminance ever seen at each cell
            self.memory = np.maximum(self.memory, Y_refl * los_scalar)
        else:
            # fully blocked: no live view, memory shows in full. Zero
            # illuminance and zero line of sight gate reflection and emission
            # off on the GPU; the memory overlay still shows because (1-los)=1.
            illuminance = 0.0
            los_scalar = 0.0

        # forget
        self.memory *= self.MEMORY_DECAY_RATE ** dt

        # Memory overlay: linear memory -> a stable display value under a FIXED
        # reference exposure (independent of live adaptation, so a remembered
        # area does not glow when the eye is dark-adapted). Reinhard + gamma.
        m = self.memory * (0.18 / MEMORY_REF_LUMINANCE)
        mem_disp = (m / (1.0 + m)) ** (1.0 / 2.2) * MEMORY_STRENGTH

        # Pack the light field: [0:3] raw linear HDR illuminance (ungated by
        # line of sight), [3] the line-of-sight scalar the GPU gates against.
        light = np.empty((h, w, 4), dtype='float32')
        light[:, :, :3] = illuminance
        light[:, :, 3] = los_scalar
        self.light.set_data(light)

        # Memory overlay: shown only where the cell is not currently in view.
        self.memory_overlay.set_data(mem_disp * (1.0 - los_scalar))

    def _composite_illuminance(self):
        """Sum this level's light maps into one HDR illuminance field (lux, RGB).

        The single place the composite is built. Snapshots the lights list (a
        spell mob may add or remove lights from its own animation thread, see
        spell.py) and sizes every map to this level, so no light on another level
        can contribute a differently-shaped array. A level holding no light
        composites to zeros (ArraySumCache.sum_arrays asserts on an empty list)."""
        lights = []
        for light in list(self.lights):
            light_map = light.lightmap(supersample=self.supersample)
            if light_map is None:
                continue
            lights.append(light_map)
        if lights:
            return self.light_cache.sum_arrays(lights).astype('float32', copy=False)
        return np.zeros(self.field_shape, dtype='float32')

    def illuminance_map(self):
        """This level's composited HDR illuminance (lux, RGB), rebuilt if dirty.

        The up-to-date map anyone may request: it recomposites when the field
        has never been built (fresh from enter()) or was marked dirty by a move
        or a light change, and otherwise returns the cached field untouched.

        Recompositing sums the light maps, which for a point light samples the
        injected shadow provider -- a GL read-back -- so it must run on the
        thread that owns that provider (the display backend's draw thread).
        Because a move only *flags* the field dirty rather than dropping it, a
        reader that just needs a value between frames (a darkness test on the
        input thread) reads the last composite through illuminance_at and never
        forces a recomposite off the draw thread."""
        if self.illuminance is None or self._illuminance_dirty:
            self.illuminance = self._composite_illuminance()
            self._illuminance_dirty = False
        return self.illuminance

    def illuminance_at(self, pos):
        """Composited HDR illuminance (lux, per channel) arriving at maze cell
        *pos* (x, y), or None if the level has not yet been composited since it
        was entered.

        Reads the cached composite without forcing a rebuild -- safe to call off
        the draw thread -- and the field is ungated by line of sight, so it
        measures the light reaching a cell regardless of the current viewpoint.
        A move leaves the field in place (only flagged dirty), so this keeps
        answering with the last composite until the draw thread recomposites via
        illuminance_map. Read the reference once so a concurrent rebuild cannot
        tear the read."""
        x, y = pos
        ss = self.supersample
        field = self.illuminance
        if field is None:
            return None
        return field[y * ss, x * ss]

    def is_dark_at(self, pos, threshold):
        """True if cell *pos* is in effectively complete darkness: the perceived
        luminance of the illuminance arriving there is below *threshold* (lux).
        A lit torch or the glo spell is one of this level's lights, so it lifts
        the player out of darkness automatically once composited.

        Returns None until the level's lighting has been composited at least once
        since it was entered (illuminance_at is None): before that there is no
        light field to judge, and a darkness warning fired then would misjudge
        the very cell the player just arrived on -- the daylit hole they dropped
        through -- as pitch dark."""
        illuminance = self.illuminance_at(pos)
        if illuminance is None:
            return None
        return float(illuminance @ LUMINANCE_WEIGHTS) < threshold

    def __repr__(self):
        return "<Level %r %dx%d>" % ((self.name,) + self.maze.shape)


class LevelPortal:
    """A join between two levels, with a :class:`~.portal.PortalEnd` per side.

    The portal only ties the two ends together and lets you step from one to
    the other. Each end is an entity already standing on its own maze (see
    :mod:`.portal`); constructing the portal just records the pairing, so the
    end and the join agree on which two mouths are connected.
    """

    def __init__(self, end_a, end_b):
        self.ends = (end_a, end_b)
        for end in self.ends:
            end.portal = self

    def other(self, end):
        """Return the end opposite *end*."""
        a, b = self.ends
        if end is a:
            return b
        if end is b:
            return a
        raise ValueError("%r is not an end of %r" % (end, self))

    def __repr__(self):
        return "<LevelPortal %r <-> %r>" % self.ends


class World:
    """Every level, the portals between them, and which level is current.

    Owns the one shared :class:`~.blocktypes.BlockTypes` table: every maze in
    the world indexes the same table, so block ids mean the same thing on every
    level and the scene's glyph registry sees each block character once.
    """

    def __init__(self, blocktypes=None):
        self.blocktypes = blocktypes if blocktypes is not None else BlockTypes()
        self.levels = {}
        self.portals = []
        self.current = None

    def add_level(self, level):
        """Add *level* to the world; the first one added becomes current."""
        assert level.name not in self.levels, "duplicate level %r" % level.name
        self.levels[level.name] = level
        level.world = self
        if self.current is None:
            self.current = level
        return level

    def add_portal(self, portal):
        """Record *portal* between its two already-placed ends."""
        self.portals.append(portal)
        return portal

    def link(self, end_a, end_b):
        """Join two :class:`~.portal.PortalEnd` entities into a portal.

        The ends are already standing on their mazes; this ties them together
        and registers the join. Returns the :class:`LevelPortal`.
        """
        return self.add_portal(LevelPortal(end_a, end_b))

    def level(self, level):
        """Resolve *level*, given as a name or a Level, to a Level."""
        return self.levels[level] if isinstance(level, str) else level

    def level_for_maze(self, maze):
        """Return the Level whose maze is *maze*, or None."""
        for level in self.levels.values():
            if level.maze is maze:
                return level
        return None

    def portal_end_at(self, level, pos):
        """Return the :class:`~.portal.PortalEnd` at *pos* on *level*, or None.

        *level* may be a name, a Level, or a Maze -- the caller usually has
        whichever of those is nearest to hand. The dungeon master finds the end
        under the player through the maze inventory instead; this is for callers
        that have a level and a cell but not the maze's occupants to hand.
        """
        if not isinstance(level, (str, Level)):
            level = self.level_for_maze(level)
        else:
            level = self.level(level)
        pos = tuple(pos)
        for portal in self.portals:
            for end in portal.ends:
                if end.level is level and end.pos == pos:
                    return end
        return None
