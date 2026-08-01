import time

from .layers import GlyphRegistry, SpriteLayer, LayerList
from .maze import Maze
from .entity import Entity
from .events import Observable
from .world import SIGHT_SUPERSAMPLE, Level
from .perception import AUTO_LOG_SALIENCE



class MessageLog(object):
    """Messages shown to the user, held as game state (``scene.log``).

    Follows the layer change-tracking contract: ``version`` bumps on every
    mutation and the ``changed`` Observable is invoked after each bump. Game
    threads write messages; a game-side painter (hud.py) renders the tail into
    a CharGridLayer — nothing rendering-side reads the log directly.
    """
    def __init__(self):
        self.lines = []
        self.version = 0
        self.changed = Observable()

    def write(self, text):
        """Append *text* to the log, splitting on newlines."""
        self.lines.extend(text.split('\n'))
        self._changed()

    def set_last_line(self, line):
        """Replace the last line (command-prompt editing)."""
        self.lines[-1] = line
        self._changed()

    def remove_last_line(self):
        self.lines.pop(-1)
        self._changed()

    def _changed(self):
        self.version += 1
        self.changed()


class Screen(object):
    """The canvas size in character cells (rows, cols), held as game state.

    The display backend writes it on window resize; game-side painters
    (hud.Hud) observe it and re-lay out their grids. Follows the layer
    change-tracking contract (``version`` + ``changed`` Observable). The
    default matches the default 1400x900 window at (10, 16)-pixel cells, so
    headless code sees the standard layout.
    """
    def __init__(self, shape=(56, 140)):
        self.shape = tuple(shape)
        self.version = 0
        self.changed = Observable()

    def set_shape(self, shape):
        """Record a new cell shape; no-op (and no bump) when unchanged."""
        shape = tuple(shape)
        if shape == self.shape:
            return
        self.shape = shape
        self.version += 1
        self.changed()


class Scene(Entity):
    """Game state: the landscape, player, items, and mobs.

    Contains no rendering code. What should be displayed is written into
    game-owned render layers (see layers.py):

    - ``glyphs`` and ``sprite_layers`` ('scenery', 'items', 'actors') carry
      character sprites; entities write into them as they move.
    - ``level.light`` (illuminance + line of sight) and
      ``level.memory_overlay`` are FieldLayers updated by
      ``Level.update_sight()``; the renderer uploads them and gates
      reflection/emission by line of sight on the GPU.

    Everything sized to a maze -- the sight fields, the composited lighting,
    and the ``visibility`` shadow provider a backend injects -- lives on the
    :class:`~.world.Level`, not on the scene. The scene reads the current
    level's fields through the delegating properties below, but the level is
    the unit of consistency: a renderer captures a level and derives every
    shape from it, so a maze and a field of the wrong size are never observed
    together. A rendering backend (e.g. backends.vispy.VispySceneRenderer)
    injects each level's ``visibility``: an object with
    ``render(pos, read=True) -> (h, w, >=3) array`` that computes a shadow map
    for a light/viewer at a maze position.
    """

    # sight memory decay now lives on the Level (which does the compositing);
    # kept here as an alias for callers that reach for scene.MEMORY_DECAY_RATE
    MEMORY_DECAY_RATE = Level.MEMORY_DECAY_RATE

    def __init__(self):
        Entity.__init__(self, entity_type='scene')
        self._player = None

        # game-owned render layers: the bridge between game state and renderers
        self.glyphs = GlyphRegistry()
        self.sprite_layers = {name: SpriteLayer(name) for name in ('scenery', 'items', 'actors')}

        # screen-space CharGridLayers (menus, pagers, console/HUD text); a
        # backend renders each entry generically, in list order
        self.grids = LayerList()

        # messages to be shown to the user; hud.ConsolePainter renders these
        self.log = MessageLog()

        # canvas size in cells, backend-written; the HUD lays out against it
        self.screen = Screen()

        # subscribed to by the display backend; see request_redraw()
        self.redraw_requested = Observable()

        # invoked (no arguments) after set_level() has swapped in a new maze
        # and resized the sight fields. Subscribers rebuild whatever they
        # sized from the old maze -- the vispy backend rebuilds its
        # ShadowRenderer and sight texture. Fired on the calling (game)
        # thread, after all of the scene's own state is consistent.
        self.level_changed = Observable()

        # set by the gameplay input handler (Escape); the display backend
        # polls it from its frame tick and shuts down
        self.quit_requested = False

        self.supersample = SIGHT_SUPERSAMPLE

        # the visible Level; set by set_level(). The maze and the sight fields
        # are read through it (see the properties below) rather than copied
        # onto the scene, so "the current level's field" is never a separate
        # fact that can fall out of step with the current level.
        self._level = None
        self._scenery = None

        # track all items
        self.items = []

        # track monsters by location
        self.monsters = {}

        # the multi-level world, installed by set_world(). Until then the scene
        # runs on a single unnamed maze, which is what the tests and the
        # screenshot harness want -- they place things at dungeon coordinates
        # and never travel.
        self.world = None

        self.clock = time.monotonic

        self.set_level(Maze.load_image('level1.png'))

    def set_world(self, world):
        """Install *world* and switch to its current level.

        The scene delegates nothing else to the world: it still holds exactly
        one visible maze. What the world adds is the ability to ask *which
        level is this* and *what portal is underfoot* (see dm.DungeonMaster).
        """
        self.world = world
        self.set_level(world.current)

    @property
    def level(self):
        """The Level currently displayed."""
        return self._level

    # The level owns these; the scene reads them through whichever level is
    # current. They are properties rather than copies so that there is exactly
    # one place each value lives -- copying them onto the scene is what let a
    # maze and a sight field of the wrong size be observed together.
    @property
    def maze(self):
        return self._level.maze

    @property
    def field_shape(self):
        return self._level.field_shape

    @property
    def memory(self):
        return self._level.memory

    @property
    def line_of_sight(self):
        return self._level.line_of_sight

    @property
    def light(self):
        """The current level's light field (rgb = illuminance, a = line of sight)."""
        return self._level.light

    @property
    def memory_overlay(self):
        """The current level's display-space memory overlay field."""
        return self._level.memory_overlay

    def set_level(self, level):
        """Make *level* visible, rebuilding everything sized from its maze.

        *level* may be a :class:`~.world.Level` or a bare Maze, which is
        wrapped in an anonymous Level -- so the scene always has a level, and
        the delegating properties above always have somewhere to point. It is
        the single entry point for level initialization: called once from
        __init__ for the starting level, and again for every later switch.

        Callers must move the player onto the new level themselves; the scene
        does not place entities. The order in which this method mutates state
        no longer matters for a concurrent draw: every maze-sized array lives
        on the level, and the renderer composites through a captured level, so
        it can never pair the new level's field with the old level's lighting.
        """
        if not isinstance(level, Level):
            level = level.level or Level(level.obj_name or 'level', level,
                                         supersample=self.supersample)

        # the outgoing level has no viewer any more, so nothing on it is in
        # sight; its memory is left alone, being what the player remembers
        if self._level is not None and self._level is not level:
            self._level.clear_line_of_sight()

        self._level = level
        if self.world is not None and level.name in self.world.levels:
            self.world.current = level

        # a light on this level announces changes to the level, which asks the
        # display to repaint through here; connect is idempotent, so a level
        # re-entered many times never stacks duplicate callbacks
        level.lighting_changed.connect(self.request_redraw)
        level.maze.appearance_changed.connect(self.request_redraw)

        # rebuild the scenery sprites: free the outgoing maze's slot before
        # allocating the new one, so the layer does not grow by a whole maze
        # on every switch.
        scenery_layer = self.sprite_layers['scenery']
        if self._scenery is not None:
            scenery_layer.remove_sprites(self._scenery)
        self._scenery = level.maze.add_scenery(self.glyphs, scenery_layer)

        # drop the level's cross-frame caches and blank its field so it is
        # built from scratch the first time it is shown
        level.enter()

        self.level_changed()

    @property
    def player(self):
        return self._player

    @player.setter
    def player(self, player):
        assert self._player is None
        self._player = player
        player.location.global_changed.connect(self._player_moved)
        player.location.global_changed.connect(self._check_nearby_percepts)
        self._player_moved()
        self._check_nearby_percepts()

    def _player_moved(self, event=None):
        # the level the player now stands on must recast sight and rescale its
        # lighting for the new viewpoint; the level the player left had its
        # sight cleared in set_level and needs nothing here
        level = self._player.level if self._player is not None else None
        if level is not None:
            level.invalidate_sight()

    def _check_nearby_percepts(self, event=None):
        """Evaluate every percept-bearing item on the player's current level
        on every player move, so detection can fire on approach rather than
        only when standing exactly on an item's cell.

        Filtering `self.items` to the current level with non-empty `.percepts`
        is cheap at this game's scale (a handful of items per level) -- no
        spatial registry needed. The actual "visibility map" lookup happens
        inside each percept's own `detectability()`, via the level's already-
        cached `illuminance_at`/`line_of_sight_at`, so the expensive part
        stays O(1) regardless of map size."""
        level = self._player.level if self._player is not None else None
        if level is None:
            return
        for item in self.items:
            loc = item.location.global_location
            if loc is None or loc.container.level is not level or not item.percepts:
                continue
            for percept in item.percepts:
                self.perceive(percept, self._player)

    def monster_moved(self, mon, old_pos):
        if old_pos is not None:
            self.monsters[tuple(old_pos)].remove(mon)
        self.monsters.setdefault(tuple(mon.position), []).append(mon)

    def add_item(self, item):
        self.items.append(item)

    def write(self, message):
        """Display a message to the user."""
        self.log.write(message)

    def perceive(self, percept, who=None):
        """Central perception resolver: decide whether *who* notices *percept*
        now, and if so deliver it. Runs a three-stage pipeline so each party
        only answers what it actually knows:

        1. ``percept.detectability(who)`` -- the percept's own physical
           detectability, per sense, to a generic observer at *who*'s
           position (lighting, camouflage, distance -- see perception.py).
        2. ``self.modify_detectability(...)`` -- environment-wide conditions
           (fog, ambient noise) the scene, not the percept, knows about.
        3. ``who.detects(...)`` -- whether *who*'s own senses clear that bar,
           and, if so, ``who.cares_about(percept)`` -- whether it is
           important enough to announce immediately rather than held for a
           later "look".

        Returns whether *who* noticed it. *who* defaults to None, meaning the
        percept is offered to anyone/everyone nearby (a noise everyone in
        earshot might catch) rather than to one actor. That broadcast is not
        implemented yet, so None currently raises; callers with a specific
        perceiver pass it explicitly.
        """
        if who is None:
            raise NotImplementedError(
                "perceive(who=None): offering a percept to everyone nearby is "
                "not implemented yet; pass an explicit perceiver")
        now = self.clock()
        if not percept.ready(now):
            return False
        raw = percept.detectability(who)
        adjusted = self.modify_detectability(raw, percept, who)
        if not who.detects(adjusted):
            return False
        percept.mark(now)
        if who.cares_about(percept):
            percept.on_perceived(self, who)
        return True

    def modify_detectability(self, values, percept, who):
        """Environment-wide modifiers (fog, ambient noise...) applied on top
        of a percept's own physical detectability, before the perceiver's
        stats are weighed. Nothing environment-wide exists in the game yet,
        so this is a pass-through -- the seam a future scene-wide condition
        would hang off."""
        return values

    def items_at(self, pos):
        """Return the items lying in the maze at *pos*."""
        return [e for e in self.maze.inventory[tuple(pos)] if e.type.isa('item')]

    def request_redraw(self):
        """Ask the display to repaint.

        Entities call this when they change something the display derives but
        does not observe directly -- lighting, most notably, which is
        recomputed during the draw itself and so cannot be an observed layer.
        Follows the observer contract used by the render layers: the backend
        subscribes to ``redraw_requested`` with a callback that only sets a
        dirty flag, so this is safe to call from any thread.
        """
        self.redraw_requested()

    def update_sight(self, dt):
        """Advance the current level's sight/memory field by *dt* seconds.

        A convenience for single-threaded callers (tests, the screenshot
        harness): it composites whichever level is current. The display
        backend does *not* go through here -- it calls ``Level.update_sight``
        on the level it has captured, so a level switch on another thread
        cannot change which level a frame draws (see the vispy renderer).
        """
        self._level.update_sight(dt, self._player)
