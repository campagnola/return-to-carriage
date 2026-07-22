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
#: gets on screen.
MEMORY_REF_LUMINANCE = 2.0
MEMORY_STRENGTH = 1.0


class Level:
    """One maze, under a name, plus everything sized against that maze.

    A Level owns *every* array sized to its maze -- the line-of-sight and
    memory fields, the composited lighting, the composited ``sight`` field the
    renderer uploads, and (injected by the display backend) the ``visibility``
    shadow provider. Nothing maze-sized lives on the scene "for whichever level
    is current"; it all lives here, on the level it belongs to.

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
        # maps, RGB), and its cross-frame cache. Sized to this maze and reached
        # only through this level, so the lights summed here (this level's) and
        # the field they multiply (this level's line of sight) can never be of
        # two different shapes. This is the expensive step; it is rebuilt only
        # when dropped (a lighting or viewpoint change), never per frame.
        #
        # There is no longer any per-frame renormalisation. The CPU emits a
        # linear HDR field and the GPU tone-maps it under a slowly-varying
        # exposure; a flickering flame therefore modulates the output directly,
        # because it changes ``illuminance`` and nothing divides that change
        # back out. (The old ``_light_norm`` divisor existed only to keep a
        # per-level log-normalisation from pinning the brightest cell to 1.0
        # and cancelling the flicker; with the tone map moved to the GPU it is
        # gone.)
        self.illuminance = None
        self.light_cache = ArraySumCache()

        # Per-cell material reflectance luminance at field resolution, (h, w, 1).
        # Built once from the fixed maze -- each block id maps to the luminance
        # of its base ``bg_color`` -- and kept, since the maze never changes. It
        # relates arriving light (illuminance) to reflected luminance for the
        # eye-adaptation target and the memory field.
        self._albedo_lum = None

        # The composited sight field the renderer uploads: RGBA float32. Owned
        # by the level so its identity is stable for a backend that captured it,
        # and always the right shape for this maze. Channels [0:3] are the
        # linear HDR visible illuminance ``los * E`` (tone-mapped on the GPU);
        # channel [3] is a display-space memory overlay, already gamma-encoded.
        self.sight = FieldLayer('sight', shape=(h, w, 4))

        # Shadow-map provider sized to this maze, injected by the display
        # backend when it builds this level's GL resources (see the vispy
        # renderer's _rebuild_for_level). Duck-typed render(pos, read=True) ->
        # (h, w, >=3) array; no rendering library is imported here to hold it.
        self.visibility = None

        # recompute line of sight on the next update (the viewer just arrived
        # or moved); set true so the first frame casts sight from scratch
        self._need_los_update = True

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
        """Discard the composited illuminance; it is rebuilt on the next update.

        Called by a light on this level whose emitted light changed (colour,
        brightness). Nothing needs to be *kept* to make a flickering flame show:
        the field is linear HDR and the GPU exposure varies slowly, so a rebuilt
        ``illuminance`` carrying the flame's new brightness modulates the output
        directly rather than being renormalised away.
        """
        self.illuminance = None

    def invalidate_sight(self):
        """The viewer moved on this level: recast sight and recomposite light.

        Drops the line of sight and the composited illuminance, so the next
        update rebuilds both for the new viewpoint.
        """
        self._need_los_update = True
        self.illuminance = None

    def enter(self):
        """Prepare this level to be shown, dropping every cross-frame cache.

        Called when the level becomes the displayed one. Blanks the composited
        field and the illuminance cache so the first frame is built from
        scratch, matching what a freshly-entered level should look like. The
        albedo map is *not* dropped: it depends only on the fixed maze.
        """
        self.illuminance = None
        self._need_los_update = True
        self.sight.set_data(np.zeros((*self.memory.shape, 4), dtype='float32'))

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
        """Advance this level's sight/memory field by *dt* seconds, writing the
        result into ``self.sight``.

        Called once per rendered frame by the display backend, for the level it
        is currently showing. Everything read here -- the line-of-sight and
        lighting fields, the lights that feed them, the shadow provider -- is
        this one level's, sized to this one maze, so no interleaving with a
        level switch on another thread can compose arrays of two shapes.

        The CPU no longer tone-maps. Channels [0:3] of ``sight`` carry the raw
        linear HDR visible illuminance ``line_of_sight * illuminance``; the GPU
        applies albedo, the Reinhard curve and display gamma under the player's
        eye-adaptation exposure. This method also drives that adaptation, from
        the reflected luminance of the blocks in a window around the player, and
        maintains the display-space memory overlay it packs into channel [3].

        When *player* is not standing on this level the view is fully blocked:
        line of sight is zero, so channels [0:3] are zero and only the memory
        overlay survives. That is what the renderer shows in the brief window
        after it has switched to a new level but before the player has been
        moved onto it -- the level's memory, for free.
        """
        watched = player is not None and player.level is self
        h, w = self.memory.shape

        if watched:
            if self._need_los_update:
                self.line_of_sight = player.line_of_sight().astype('float32', copy=False)
                self._need_los_update = False
            line_of_sight = self.line_of_sight

            # Composite this level's HDR illuminance. Only this level's lights
            # are consulted, so their maps are all sized to this level -- no
            # light on another level can contribute a differently-shaped array.
            # Held in a local because an animator/flicker thread may null the
            # cache at any moment; the worst that costs is one stale frame.
            illuminance = self.illuminance
            if illuminance is None:
                lights = []
                # snapshot: a spell mob may add or remove lights from its own
                # animation thread while this composite runs (see spell.py), so
                # iterate a copy rather than the live list
                for light in list(self.lights):
                    light_map = light.lightmap(supersample=self.supersample)
                    if light_map is None:
                        continue
                    lights.append(light_map)
                if lights:
                    # ArraySumCache.sum_arrays asserts on an empty list, and a
                    # level may legitimately hold no light at all
                    illuminance = self.light_cache.sum_arrays(lights).astype('float32', copy=False)
                else:
                    illuminance = np.zeros(self.field_shape, dtype='float32')
                self.illuminance = illuminance
            if self._albedo_lum is None:
                self._albedo_lum = self._build_albedo_lum()

            # linear HDR reflected-light input for the GPU tone map
            E_vis = line_of_sight * illuminance
            los_scalar = line_of_sight.max(axis=2)

            # Reflected luminance per cell: what the eye and memory respond to.
            lumE = illuminance @ LUMINANCE_WEIGHTS
            Y_refl = self._albedo_lum[:, :, 0] * lumE

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
            # fully blocked: no live view, memory shows in full
            line_of_sight = 0.0
            E_vis = 0.0
            los_scalar = 0.0

        # forget
        self.memory *= self.MEMORY_DECAY_RATE ** dt

        # Memory overlay: linear memory -> a stable display value under a FIXED
        # reference exposure (independent of live adaptation, so a remembered
        # area does not glow when the eye is dark-adapted). Reinhard + gamma.
        m = self.memory * (0.18 / MEMORY_REF_LUMINANCE)
        mem_disp = (m / (1.0 + m)) ** (1.0 / 2.2) * MEMORY_STRENGTH

        # Pack RGBA: [0:3] linear HDR visible light, [3] memory where not seen.
        out = np.empty((h, w, 4), dtype='float32')
        out[:, :, :3] = E_vis
        out[:, :, 3] = mem_disp * (1.0 - los_scalar)
        self.sight.set_data(out)

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
