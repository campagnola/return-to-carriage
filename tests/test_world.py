"""Levels, portals, and travelling between them."""
import numpy as np
import pytest

from carriage_return.blocktypes import BlockTypes
from carriage_return.dm import DungeonMaster
from carriage_return.levels import build_world, level_002_sewer
from carriage_return.maze import Maze
from carriage_return.player import Player
from carriage_return.portal import Door, Hole, StairsDown, StairsUp
from carriage_return.scene import Scene
from carriage_return.world import Level, LevelPortal, World


def _flat_world(scene):
    """Two bare levels joined by a one-way hole, with nothing else on them.

    Portal ends are entities, so they want the *scene* they will draw on -- the
    caller creates it first and passes it in.
    """
    world = World()
    bt = world.blocktypes
    for name in ('upper', 'lower'):
        maze = Maze.filled((10, 10), bt, 'path', obj_name=name)
        world.add_level(Level(name, maze))
    top = Hole(location=(world.levels['upper'].maze, (3, 3)), scene=scene)
    bottom = Hole(location=(world.levels['lower'].maze, (6, 6)), scene=scene,
                  enterable=False)
    world.link(top, bottom)
    return world


class FakeVisibility:
    """Everything is visible. Sized per call: field_shape changes with level."""
    def __init__(self, scene):
        self.scene = scene

    def render(self, pos, read=True):
        return np.full(self.scene.field_shape[:2] + (4,), 255, dtype='ubyte')


def _auto_visibility(scene):
    """Inject a FakeVisibility onto every level the scene shows, as the real
    renderer does on level_changed -- so a test that travels between levels
    gets each one's shadow provider without wiring it up by hand."""
    scene.level_changed.connect(lambda: setattr(scene.level, 'visibility',
                                                 FakeVisibility(scene)))
    scene.level.visibility = FakeVisibility(scene)


@pytest.fixture
def played_world():
    """A scene running _flat_world(), with a player on the upper level."""
    scene = Scene()
    _auto_visibility(scene)
    world = _flat_world(scene)
    scene.set_world(world)
    player = Player(scene)
    player.location.update(world.levels['upper'].maze, (1, 1))
    return scene, world, player, DungeonMaster(scene)


# -- portal structure ---------------------------------------------------------

def test_portal_ends_stand_on_their_cells_as_entities():
    """An end is an entity in its maze's inventory, not a stamped block."""
    scene = Scene()
    world = _flat_world(scene)
    upper = world.levels['upper'].maze
    lower = world.levels['lower'].maze
    top, bottom = world.portals[0].ends

    assert top in upper.inventory[(3, 3)]
    assert bottom in lower.inventory[(6, 6)]
    # the floor underneath is untouched: portals no longer stamp terrain
    assert upper.blocktype_at(3, 3)['name'] == 'path'


def test_each_kind_of_end_knows_how_it_is_operated():
    """Behaviour is a class default of the kind, so placing one respells nothing."""
    assert StairsDown.command == '>' and StairsDown.walk_on is False
    assert StairsUp.command == '<' and StairsUp.walk_on is False
    assert Hole.command is None and Hole.walk_on is True
    assert Door.command is None and Door.walk_on is True


def test_other_returns_the_opposite_end():
    scene = Scene()
    world = _flat_world(scene)
    portal = world.portals[0]
    a, b = portal.ends
    assert portal.other(a) is b
    assert portal.other(b) is a


def test_other_rejects_a_foreign_end():
    scene = Scene()
    world = _flat_world(scene)
    stranger = Door(location=(world.levels['upper'].maze, (0, 0)), scene=scene)
    with pytest.raises(ValueError):
        world.portals[0].other(stranger)


def test_portal_end_at_finds_ends_by_level_name_or_maze():
    scene = Scene()
    world = _flat_world(scene)
    upper = world.levels['upper']
    assert isinstance(world.portal_end_at('upper', (3, 3)), Hole)
    assert world.portal_end_at(upper, (3, 3)) is not None
    assert world.portal_end_at(upper.maze, (3, 3)) is not None
    assert world.portal_end_at(upper, (9, 9)) is None
    # same position, different level: the ends must not be confused
    assert world.portal_end_at('lower', (3, 3)) is None


def test_duplicate_level_names_are_rejected():
    world = World()
    maze = Maze.filled((4, 4), world.blocktypes, 'path')
    world.add_level(Level('a', maze))
    with pytest.raises(AssertionError):
        world.add_level(Level('a', maze))


# -- travelling ---------------------------------------------------------------

def test_walking_onto_a_hole_changes_level(played_world):
    scene, world, player, dm = played_world

    dm.move_player(player, np.array([3, 3]))

    assert scene.level is world.levels['lower']
    assert scene.maze is world.levels['lower'].maze
    assert tuple(player.location.slot) == (6, 6)


def test_a_one_way_portal_refuses_the_return_trip(played_world):
    scene, world, player, dm = played_world
    dm.move_player(player, np.array([3, 3]))
    lower_end = world.portal_end_at('lower', (6, 6))

    assert dm.request_traverse(player, lower_end) is False
    assert scene.level is world.levels['lower']
    assert 'no way back up' in scene.log.lines[-1]


def test_stairs_need_the_matching_command():
    scene = Scene()
    _auto_visibility(scene)
    world = World()
    bt = world.blocktypes
    for name in ('top', 'bottom'):
        world.add_level(Level(name, Maze.filled((10, 10), bt, 'path', obj_name=name)))
    world.link(StairsDown(location=(world.levels['top'].maze, (4, 4)), scene=scene),
               StairsUp(location=(world.levels['bottom'].maze, (2, 2)), scene=scene))
    scene.set_world(world)
    player = Player(scene)
    dm = DungeonMaster(scene)

    # walking onto stairs does nothing on its own
    dm.move_player(player, np.array([4, 4]))
    assert scene.level is world.levels['top']

    # ...and neither does the wrong command
    dm.use_stairs(player, '<')
    assert scene.level is world.levels['top']

    dm.use_stairs(player, '>')
    assert scene.level is world.levels['bottom']
    assert tuple(player.location.slot) == (2, 2)


def test_use_stairs_away_from_any_portal_says_so(played_world):
    scene, world, player, dm = played_world

    dm.use_stairs(player, '>')

    assert scene.level is world.levels['upper']
    assert 'no stairs down' in scene.log.lines[-1]


def test_changing_level_resizes_the_sight_fields(played_world):
    scene, world, player, dm = played_world
    world.add_level(Level('big', Maze.filled((20, 30), world.blocktypes, 'path')))
    ss = scene.supersample

    scene.set_level(world.levels['big'])

    assert scene.field_shape == (20 * ss, 30 * ss, 3)
    assert scene.memory.shape == scene.field_shape[:2]
    assert scene.line_of_sight.shape == scene.field_shape
    assert scene.light.data.shape == scene.field_shape[:2] + (4,)
    assert scene.memory_overlay.data.shape == scene.field_shape[:2]


def test_entities_on_other_levels_are_hidden(played_world):
    """Sprite layers are shared by the world, so off-level things must hide."""
    from carriage_return.item import Scroll

    scene, world, player, dm = played_world
    scroll = Scroll(location=(world.levels['upper'].maze, (2, 2)), scene=scene)
    assert not np.isnan(scroll.sprite.sprite.position).any()

    dm.move_player(player, np.array([3, 3]))   # down the hole to 'lower'

    assert np.isnan(scroll.sprite.sprite.position).all()
    assert not np.isnan(player.sprite.sprite.position).any()

    scene.set_level(world.levels['upper'])
    assert not np.isnan(scroll.sprite.sprite.position).any()


def test_a_light_registers_with_the_level_it_stands_on(played_world):
    from carriage_return.item import Torch

    scene, world, player, dm = played_world
    upper, lower = world.levels['upper'], world.levels['lower']
    torch = Torch(location=(upper.maze, (2, 2)), scene=scene)

    assert torch.light in upper.lights
    assert torch.light not in lower.lights

    torch.location.update(lower.maze, (4, 4))

    assert torch.light not in upper.lights
    assert torch.light in lower.lights


def test_a_carried_light_follows_its_bearer_between_levels(played_world):
    from carriage_return.item import Torch

    scene, world, player, dm = played_world
    upper, lower = world.levels['upper'], world.levels['lower']
    torch = Torch(location=(player, 'right hand'), scene=scene)
    assert torch.light in upper.lights

    dm.move_player(player, np.array([3, 3]))   # down the hole to 'lower'

    assert torch.light not in upper.lights
    assert torch.light in lower.lights


def test_a_map_light_shines_from_a_fixed_cell(played_world):
    """A light attached to the map (a hole in the ceiling) registers with its
    level, contributes a map sized to that level, and does not move when the
    player does."""
    from carriage_return.light import PointLight

    scene, world, player, dm = played_world
    upper, lower = world.levels['upper'], world.levels['lower']

    light = upper.maze.add_light(PointLight(upper.maze, color=(9, 9, 9)), pos=(5, 5))
    assert light in upper.lights
    assert light not in lower.lights
    assert light.global_place() == (upper.maze, (5, 5))

    lm = light.lightmap(supersample=upper.supersample)
    assert lm.shape == upper.field_shape

    # the player walking away leaves a map light exactly where it was pinned
    dm.move_player(player, np.array([3, 3]))   # down the hole to 'lower'
    assert light in upper.lights
    assert light.global_place() == (upper.maze, (5, 5))


def test_a_light_on_another_level_does_not_light_this_one(played_world):
    """Only the current level's lights are composited, so a light elsewhere
    cannot contribute (nor contribute a wrongly-shaped map)."""
    from carriage_return.item import Torch

    scene, world, player, dm = played_world
    upper, lower = world.levels['upper'], world.levels['lower']
    Torch(location=(upper.maze, (2, 2)), scene=scene)

    scene.set_level(lower)
    scene.update_sight(1 / 60.)

    assert lower.lights == []
    assert not scene.light.data.any()
    assert not scene.memory_overlay.data.any()


def test_leaving_a_level_clears_its_line_of_sight(played_world):
    """Nothing is in sight where the player is not, which is also what stops
    the flicker thread burning torches nobody is watching."""
    from carriage_return.item import Torch

    scene, world, player, dm = played_world
    upper, lower = world.levels['upper'], world.levels['lower']
    torch = Torch(location=(upper.maze, (2, 2)), scene=scene)
    scene.update_sight(1 / 60.)
    assert torch.light.in_player_sight()

    scene.set_level(lower)

    assert not upper.line_of_sight.any()
    assert not torch.light.in_player_sight()


def test_compositing_a_level_the_player_has_left_uses_its_own_shape():
    """Regression: entering a differently-shaped level must not compose the new
    level's fields against the old level's lighting.

    Reproduces the crash seen entering the dungeon from the sewer -- the draw
    thread composited a level while the input thread was still moving the
    player onto it. Because update_sight is a function of one level and the
    player is not (yet) on it, the view is fully blocked: the level's own
    (correctly shaped) memory is what shows, with no broadcast against a
    field of the other level's shape.
    """
    world = World()
    bt = world.blocktypes
    world.add_level(Level('small', Maze.filled((8, 9), bt, 'path', obj_name='small')))
    world.add_level(Level('big', Maze.filled((20, 30), bt, 'path', obj_name='big')))
    small, big = world.levels['small'], world.levels['big']

    scene = Scene()
    _auto_visibility(scene)
    scene.set_world(world)                 # current == 'small' (added first)
    player = Player(scene)
    player.location.update(small.maze, (1, 1))

    big.memory[:] = 0.25                   # something remembered on the big level

    # the renderer's situation mid-transition: showing 'big' while the player
    # is still on 'small'. This used to raise "operands could not be broadcast
    # together" when the big level's fields were composited against the small
    # level's lighting; each level now derives every shape from itself.
    big.update_sight(1 / 60., player)

    # light packs RGBA ([0:3] illuminance, [3] line of sight); memory_overlay
    # is the display-space overlay. The player is elsewhere, so the view is
    # fully blocked: no current light and zero line of sight, and only the
    # level's OWN (correctly shaped) memory shows through.
    assert big.light.data.shape == big.field_shape[:2] + (4,)
    assert not big.light.data.any()            # fully blocked -> no light, no line of sight
    assert big.memory_overlay.data.shape == big.field_shape[:2]
    assert big.memory_overlay.data.any()       # its own remembered field shows


def test_a_level_keeps_its_own_memory(played_world):
    """Memory is a fact about a level, so it survives going away and back."""
    scene, world, player, dm = played_world
    upper, lower = world.levels['upper'], world.levels['lower']
    upper.memory[:] = 0.5

    scene.set_level(lower)
    assert not lower.memory.any()

    scene.set_level(upper)
    assert (upper.memory == 0.5).all()


def test_sight_fields_are_sized_to_their_own_level(played_world):
    """The invariant the whole design rests on: a level's field always matches
    that level's maze, whichever level the scene is showing."""
    scene, world, player, dm = played_world
    world.add_level(Level('big', Maze.filled((20, 30), world.blocktypes, 'path')))

    for level in world.levels.values():
        ss = level.supersample
        expected = (level.maze.shape[0] * ss, level.maze.shape[1] * ss, 3)
        assert level.field_shape == expected
        assert level.line_of_sight.shape == expected
        assert level.memory.shape == expected[:2]


def test_an_unlit_level_renders_dark_rather_than_failing(played_world):
    """No light anywhere is a legal state; it must not blow up the sum cache."""
    scene, world, player, dm = played_world

    scene.update_sight(1 / 60.)

    # the level is in view (line of sight, channel 3, is non-zero) but unlit:
    # no illuminance in the rgb channels and nothing remembered, so the GPU
    # renders it dark rather than the sum cache blowing up.
    assert scene.light.data.shape == scene.field_shape[:2] + (4,)
    assert not scene.light.data[..., :3].any()
    assert not scene.memory_overlay.data.any()


# -- the shipped levels -------------------------------------------------------

def test_home_is_a_walled_room_with_a_hole():
    world = build_world(Scene())
    home = world.levels['home']
    maze = home.maze
    hole = home.locations['hole']

    # the hole is a portal entity placed by the sewer builder, not terrain here
    assert maze.blocktype_at(hole[1], hole[0])['name'] == 'path'
    border = np.concatenate([maze.blocks[0], maze.blocks[-1],
                             maze.blocks[:, 0], maze.blocks[:, -1]])
    assert (border == maze.blocktypes.id_of('wall')).all()
    assert (maze.blocks[1:-1, 1:-1] == maze.blocktypes.id_of('path')).all()


def test_the_sewer_is_the_same_every_time():
    a_maze, a_loc, _ = level_002_sewer._generate(BlockTypes())
    b_maze, b_loc, _ = level_002_sewer._generate(BlockTypes())

    assert (a_maze.blocks == b_maze.blocks).all()
    assert a_loc == b_loc


def test_the_sewer_hallways_join_up():
    """The stairs must be walkable-reachable from the hole, for any seed."""
    for seed in (20240719, 1, 2, 3, 99):
        maze, loc, _ = level_002_sewer._generate(BlockTypes(), seed=seed)
        hole, stairs = loc['hole'], loc['stairs_down']
        walkable = maze.blocktypes['walkable'][maze.blocks]
        seen = np.zeros(maze.shape, dtype=bool)
        stack = [(hole[1], hole[0])]
        seen[stack[0]] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (0 <= ny < maze.shape[0] and 0 <= nx < maze.shape[1]
                        and walkable[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        assert seen[stairs[1], stairs[0]], "seed %s: stairs unreachable" % seed


def test_the_sewer_stays_inside_its_walls():
    for seed in (20240719, 1, 2, 3, 99):
        maze, _, _ = level_002_sewer._generate(BlockTypes(), seed=seed)
        walkable = maze.blocktypes['walkable'][maze.blocks]
        assert not walkable[0].any() and not walkable[-1].any()
        assert not walkable[:, 0].any() and not walkable[:, -1].any()


def test_build_world_wires_the_three_levels():
    world = build_world(Scene())

    assert set(world.levels) == {'home', 'sewer', 'dungeon'}
    assert world.current is world.levels['home']

    hole_home, hole_sewer = world.portals[0].ends
    assert hole_home.level.name == 'home' and hole_home.enterable
    assert hole_sewer.level.name == 'sewer' and not hole_sewer.enterable

    down, up = world.portals[1].ends
    assert (down.level.name, down.command) == ('sewer', '>')
    assert (up.level.name, up.command) == ('dungeon', '<')
    assert up.pos == (7, 7)


def test_every_level_shares_one_blocktype_table():
    """Block ids must mean the same thing on every level (and glyphs register once)."""
    world = build_world(Scene())
    for level in world.levels.values():
        assert level.maze.blocktypes is world.blocktypes
