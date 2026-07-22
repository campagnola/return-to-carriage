# coding: utf8
import numpy as np

from .entity import Component
from .events import Observable


class Light(Component):
    """A light source attached to an entity.

    A Light is a component like Location or Inventory: it belongs to a host
    entity -- an item, a mob, or a maze -- and is reached as ``entity.light``.
    The base class owns everything common to *being a light*, whatever the
    light looks like: membership in the right level's ``lights`` list, knowing
    which cell (and so which level) it stands on, following its host from one
    level to another, its emitted colour and brightness, and caching the
    coloured map the scene composites each frame.

    What the base does *not* decide is the *shape* of the light -- the pattern
    of brightness it paints over the map. That is the one thing subclasses
    provide, by implementing :meth:`_render_light_map`:

    - :class:`PointLight` -- an omnidirectional source that casts shadows and
      falls off as 1/r^2 from its cell (a torch, a glowing mob).
    - :class:`ArrayLight` -- an arbitrary per-cell pattern supplied as an array
      (sunlight dappled through leaves, a glowing glyph).
    - :class:`AmbientLight` -- one colour applied equally to every cell (flat
      ambient fill, the even wash of open daylight).

    Where a light *is* comes from its host. An entity-borne light (a carried
    torch, a glowing mob) tracks its host's ``location``, so it shines wherever
    the host goes and re-registers with whatever level the host walks onto. A
    light can instead be pinned to a fixed maze cell with :meth:`Maze.add_light`
    -- this is how a shaft of daylight from a hole in the ceiling is expressed:
    it belongs to the map and stays put whatever walks beneath it.
    """

    def __init__(self, entity, color=(10, 10, 10), brightness=1.0):
        Component.__init__(self, entity, component_type='light')
        self._color = tuple(color)
        self._brightness = brightness

        # Fired whenever this light's appearance changes (colour, brightness).
        # The level this light stands on subscribes (see Level.add_light) and
        # decides what to do with it -- drop its composited lighting and ask
        # the display to repaint. Announcing the change this way is what frees a
        # light from needing any reference to the scene.
        self.changed = Observable()

        # (maze, slot) when this light is pinned to a fixed cell rather than
        # following its host's location; set by pin()/Maze.add_light.
        self._fixed_place = None

        # The final coloured map the scene composites, sized to the level. It
        # is the one cache the base owns. A subclass with an expensive
        # position-dependent intermediate (a shadow map, an upsampled pattern)
        # holds that separately, so a mere colour or brightness change -- which
        # happens every frame for a flickering flame -- rescales this cheap
        # final map without rebuilding the expensive part underneath.
        self._light_map = None

        # the Level whose lights list currently holds this light
        self._light_level = None

        # A host-borne light re-registers and drops its caches every time the
        # host's global location changes -- walking, or being carried to
        # another level. A pinned light never moves, and its host (a maze) is
        # stationary, so for one of those this simply never fires.
        entity.location.global_changed.connect(self._host_moved)
        self._register()

    def pin(self, maze, pos):
        """Fix this light at cell *pos* ``(x, y)`` of *maze*.

        Used for light that belongs to the map rather than to anything that
        moves -- light falling through a hole in the ceiling. The host entity
        is the maze; the position is this cell, once and for all. Reached
        through :meth:`Maze.add_light`.
        """
        self._fixed_place = (maze, tuple(pos))
        self._invalidate_position()
        self._register()

    def global_place(self):
        """This light's ``(maze, slot)``, or None if it is nowhere.

        A pinned light reports its fixed cell; otherwise the light sits where
        its host does, read as one ``(container, slot)`` tuple so the maze and
        the slot always belong together even while the host is being moved.
        """
        if self._fixed_place is not None:
            return self._fixed_place
        ml = self.parent_entity.location.global_location
        return None if ml is None else ml.place

    @property
    def level(self):
        """The Level this light stands on, or None if it is nowhere."""
        place = self.global_place()
        if place is None:
            return None
        maze, _slot = place
        return None if maze is None else maze.level

    def _host_moved(self, event):
        self._invalidate_position()
        self._register()

    def _invalidate_position(self):
        """Drop the caches that depend on where the light is.

        The base owns only the final coloured map; a subclass whose
        intermediate maps are positioned against the maze extends this to drop
        those too (see :class:`PointLight`, :class:`ArrayLight`).
        """
        self._light_map = None

    def _register(self):
        """Keep this light in the ``lights`` list of the level it is on.

        A carried torch moves with its bearer, so this runs on every host
        location change, including the ones caused by the bearer travelling to
        another level.
        """
        level = self.level
        if level is self._light_level:
            return
        if self._light_level is not None:
            self._light_level.remove_light(self)
        self._light_level = level
        if level is not None:
            level.add_light(self)

    def destroy(self):
        """Take this light off its level; its host is going away."""
        if self._light_level is not None:
            self._light_level.remove_light(self)
            self._light_level = None

    @property
    def color(self):
        """The light's emitted colour; the host sets this.

        Changing it discards only the colour scaling of the cached light map --
        any position-dependent map underneath survives -- then announces the
        change, which the level turns into stale lighting and a repaint.
        """
        return self._color

    @color.setter
    def color(self, value):
        value = tuple(value)
        if value == self._color:
            return
        self._color = value
        self._light_map = None
        self.changed()

    @property
    def brightness(self):
        """Scale applied to ``color``; 1.0 is the light's nominal output.

        Setting it discards only the colour scaling of the cached light map --
        any position-dependent map underneath survives -- then announces the
        change, which the level turns into stale lighting and a repaint.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value):
        if value == self._brightness:
            return
        self._brightness = value
        self._light_map = None
        self.changed()

    def _scaled_color(self):
        """The emitted colour scaled by brightness, as a float32 ``(3,)`` array.

        float32 throughout: these maps are rescaled and composited every frame
        for flickering lights, and float64 doubles that cost.
        """
        return np.array(self._color, dtype='float32') * self._brightness

    def in_player_sight(self):
        """True if the player can currently see the cell this light occupies.

        Sampled from *this light's own level*, which is what makes it safe to
        call from an animator thread: the field reached through
        ``level.line_of_sight`` is sized to ``level.maze`` by construction, so
        the light's coordinates always index it, whatever the scene is showing
        at the moment.

        A level the player has left has its line of sight cleared, so lights
        there correctly report False.
        """
        place = self.global_place()
        if place is None:
            return False
        # one read: the maze and the position in it must be the same instant,
        # or a move between levels yields the new maze at the old coordinates
        maze, slot = place
        if maze is None or maze.level is None:
            return False
        level = maze.level
        x, y = slot
        ss = level.supersample
        return level.line_of_sight[y * ss, x * ss].max() > 0

    def lightmap(self, supersample=1):
        """This light's contribution, sized to the maze it is standing on.

        The shape comes from the light's own maze, so the map it returns always
        matches the level that light is on. Level.update_sight only sums that
        level's own lights, so those maps are all the same shape. Returns
        None when the light is nowhere (its host is outside any maze).
        """
        place = self.global_place()
        if place is None:
            return None
        maze, slot = place        # one read; see in_player_sight
        if maze is None:
            return None

        # Held in a local because an animator thread may null the cache at any
        # moment -- the worst that costs is one frame at the previous colour.
        light_map = self._light_map
        if light_map is None:
            light_map = self._render_light_map(maze, slot, supersample)
            self._light_map = light_map
        return light_map

    def _render_light_map(self, maze, slot, supersample):
        """Build this light's coloured map on *maze* at *slot*.

        Subclass hook. Returns a ``(h, w, 3)`` float32 array the size of the
        level's field (``maze.shape * supersample``). Called by
        :meth:`lightmap` only when the cached map is stale.
        """
        raise NotImplementedError("Light is abstract; use a PointLight, "
                                  "ArrayLight, or AmbientLight")


class PointLight(Light):
    """An omnidirectional point source: casts shadows, falls off as 1/r^2.

    This is the light of a torch or a glowing mob. Its map is the product of a
    shadow map cast from the light's cell (computed by the scene's visibility
    provider) and a 1/r^2 distance falloff, tinted by the light's colour.
    Moving the light throws both away; a colour or brightness change keeps them
    and only rescales the cheap final map, which is what lets a flame flicker
    without recasting shadows every frame.
    """

    def __init__(self, entity, color=(1, 1, 1), brightness=1.0):
        # Position-dependent caches, dropped together whenever the light moves
        # (see _invalidate_position). The shadow map and the unscaled
        # (shadow * falloff) map survive a colour or brightness change; only
        # the base's final map is rebuilt for those.
        self._shadow_map = None
        self._unscaled_light_map = None
        Light.__init__(self, entity, color=color, brightness=brightness)

    def _invalidate_position(self):
        self._shadow_map = None
        self._unscaled_light_map = None
        Light._invalidate_position(self)

    def set_shadow_map(self, smap):
        self._shadow_map = smap
        self._unscaled_light_map = None
        self._light_map = None

    def shadow_map(self, slot):
        if self._shadow_map is None:
            smap = self.level.visibility.render(slot, read=True)[..., :3]
            self.set_shadow_map(smap)
            assert self._shadow_map is not None
        return self._shadow_map

    def _render_light_map(self, maze, slot, supersample):
        # Held in a local because an animator thread may null the cache at any
        # moment -- the worst that costs is one frame at the previous brightness.
        unscaled = self._unscaled_light_map
        if unscaled is None:
            (x, y) = slot
            maze_shape = maze.shape
            maze_pos = np.mgrid[0:maze_shape[0]*supersample, 0:maze_shape[1]*supersample].transpose(1, 2, 0)
            light_pos = np.array([[[y * supersample, x * supersample]]]) + (0.5 * supersample)
            dist2 = ((maze_pos - light_pos) ** 2).sum(axis=2) + 0.5  # 0.5 enforces height
            dist2 = dist2.astype('float32')
            unscaled = self.shadow_map(slot) / dist2[:, :, None]
            self._unscaled_light_map = unscaled
        return unscaled * self._scaled_color()[None, None, :]


class ArrayLight(Light):
    """A light whose per-cell brightness is given by a fixed array.

    The array is sized to the maze -- one value per cell -- and read as a scale
    on the light's colour, so it paints an arbitrary pattern of light and shade
    that a point source cannot: sunlight dappled through leaves, a glowing
    glyph, a patch of phosphorescent moss. It casts no shadows and does not
    fall off with distance; the array *is* the shape of the light.

    *array* has shape ``(maze_h, maze_w)`` for one intensity per cell, or
    ``(maze_h, maze_w, 3)`` to scale each colour channel independently. It is
    upsampled to the level's field resolution once and cached; a colour or
    brightness change only rescales the result.
    """

    def __init__(self, entity, array, color=(10, 10, 10), brightness=1.0):
        # The caller's pattern, at maze resolution. Upsampled lazily to the
        # field resolution in _base_map and dropped if the light changes maze.
        self._array = np.asarray(array, dtype='float32')
        self._base_map = None
        Light.__init__(self, entity, color=color, brightness=brightness)

    def _invalidate_position(self):
        self._base_map = None
        Light._invalidate_position(self)

    def _render_light_map(self, maze, slot, supersample):
        base = self._base_map
        if base is None:
            arr = self._array
            assert arr.shape[:2] == tuple(maze.shape[:2]), (
                "ArrayLight array shape %r does not match maze %r"
                % (arr.shape[:2], tuple(maze.shape[:2])))
            # nearest-neighbour upsample from maze cells to field cells, the
            # same maze->field scaling the point light's mgrid produces
            base = np.repeat(np.repeat(arr, supersample, axis=0), supersample, axis=1)
            if base.ndim == 2:
                base = base[:, :, None]
            self._base_map = base
        return base * self._scaled_color()[None, None, :]


class AmbientLight(Light):
    """A constant colour applied equally to every cell of the level.

    It has a colour and brightness but no shape and no meaningful position: it
    fills the whole map uniformly -- flat ambient fill, or the even wash of
    open daylight. Add it to a level with :meth:`Maze.add_light` like any
    pinned light; its host is the maze, so it lives and dies with that level,
    but the cell it is pinned to makes no difference to what it paints.
    """

    def _render_light_map(self, maze, slot, supersample):
        shape = (maze.shape[0] * supersample, maze.shape[1] * supersample, 3)
        light_map = np.empty(shape, dtype='float32')
        light_map[:] = self._scaled_color()
        return light_map
