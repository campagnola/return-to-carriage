# coding: utf8
import numpy as np

from .entity import Component


class Light(Component):
    """A light source attached to an entity.

    A Light is a component like Location or Inventory: it belongs to a host
    entity -- an item, a mob, or a maze -- and is reached as ``entity.light``.
    It owns everything about *being a light*: membership in the right level's
    ``lights`` list, the shadow map cast from the light's cell, the distance
    falloff, and the coloured light map the scene composites each frame. The
    host decides only the light's *character* -- its colour and brightness. For
    now every light is omnidirectional.

    Where a light *is* comes from its host. An entity-borne light (a carried
    torch, a glowing mob) tracks its host's ``location``, so it shines wherever
    the host goes and re-registers with whatever level the host walks onto. A
    light can instead be pinned to a fixed maze cell with :meth:`Maze.add_light`
    -- this is how a shaft of daylight from a hole in the ceiling is expressed:
    it belongs to the map and stays put whatever walks beneath it.
    """

    def __init__(self, entity, scene, color=(10, 10, 10), brightness=1.0):
        Component.__init__(self, entity, component_type='light')
        self.scene = scene
        self._color = tuple(color)
        self._brightness = brightness

        # (maze, slot) when this light is pinned to a fixed cell rather than
        # following its host's location; set by pin()/Maze.add_light.
        self._fixed_place = None

        # cached lighting, all dropped whenever the light moves. The shadow map
        # and distance falloff survive a colour or brightness change; only the
        # final coloured map is rebuilt for those (see brightness/color).
        self._shadow_map = None
        self._unscaled_light_map = None
        self._light_map = None

        # the Level whose lights list currently holds this light
        self._light_level = None

        # A host-borne light re-registers and drops its position caches every
        # time the host's global location changes -- walking, or being carried
        # to another level. A pinned light never moves, and its host (a maze)
        # is stationary, so for one of those this simply never fires.
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
        self._shadow_map = None
        self._unscaled_light_map = None
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
            self._light_level.lights.remove(self)
        self._light_level = level
        if level is not None:
            level.lights.append(self)

    def destroy(self):
        """Take this light off its level; its host is going away."""
        if self._light_level is not None:
            self._light_level.lights.remove(self)
            self._light_level = None

    @property
    def color(self):
        """The light's emitted colour; the host sets this.

        Changing it discards only the colour scaling of the cached light map --
        the shadow map and distance falloff underneath survive -- then tells
        the scene its lighting is stale and asks the display to repaint.
        """
        return self._color

    @color.setter
    def color(self, value):
        value = tuple(value)
        if value == self._color:
            return
        self._color = value
        self._light_map = None
        self.scene.invalidate_lighting()
        self.scene.request_redraw()

    @property
    def brightness(self):
        """Scale applied to ``color``; 1.0 is the light's nominal output.

        Setting it discards only the colour scaling of the cached light map --
        the shadow map and distance falloff underneath survive -- then tells
        the scene its lighting is stale and asks the display to repaint.
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value):
        if value == self._brightness:
            return
        self._brightness = value
        self._light_map = None
        self.scene.invalidate_lighting()
        self.scene.request_redraw()

    def set_shadow_map(self, smap):
        self._shadow_map = smap
        self._unscaled_light_map = None
        self._light_map = None

    def shadow_map(self, slot):
        if self._shadow_map is None:
            smap = self.scene.visibility.render(slot, read=True)[..., :3]
            self.set_shadow_map(smap)
            assert self._shadow_map is not None
        return self._shadow_map

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
        matches the level that light is on. Scene.update_sight only sums the
        current level's lights, so those maps are all the same shape.
        """
        place = self.global_place()
        if place is None:
            return None
        maze, slot = place        # one read; see in_player_sight
        if maze is None:
            return None

        if self._unscaled_light_map is None:
            (x, y) = slot

            maze_shape = maze.shape
            maze_pos = np.mgrid[0:maze_shape[0]*supersample, 0:maze_shape[1]*supersample].transpose(1, 2, 0)
            light_pos = np.array([[[y * supersample, x * supersample]]]) + (0.5 * supersample)
            dist2 = ((maze_pos - light_pos) ** 2).sum(axis=2) + 0.5  # 0.5 enforces height
            dist2 = dist2.astype('float32')

            # float32 throughout: these maps are rescaled and composited every
            # frame for flickering lights, and float64 doubles that cost
            self._unscaled_light_map = self.shadow_map(slot) / dist2[:, :, None]

        # The shadow map and distance falloff above are cached until the light
        # moves; a brightness change only rescales that cached map. Held in a
        # local because an animator thread may null the cache at any moment --
        # the worst that costs is one frame at the previous brightness.
        light_map = self._light_map
        if light_map is None:
            color = np.array(self._color, dtype='float32') * self._brightness
            light_map = self._unscaled_light_map * color[None, None, :]
            self._light_map = light_map

        return light_map
