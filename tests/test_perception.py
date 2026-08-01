"""The perception framework and its grue client (headless).

These tests never touch real time or real randomness. ``scene.clock`` is pinned
to a controlled value, and the noticing roll is made deterministic by giving the
perceiver a fixed ``perception`` -- a StubPerceiver with a constant, or a real
Player whose ``base_perception`` distribution is replaced by a constant callable
(the seam Player.perception samples through). Nothing here opens a GUI window:
the darkness and grue paths are exercised by setting ``level.illuminance``
directly rather than rendering.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from carriage_return.blocktypes import BlockTypes
from carriage_return.dm import DungeonMaster
from carriage_return.item import Sword
from carriage_return.levels import build_world
from carriage_return.maze import Maze
from carriage_return.perception import ContinuousPercept, Percept, TransientPercept, VisualPercept
from carriage_return.player import Player
from carriage_return.random import RandomDist
from carriage_return.scene import AUTO_LOG_SALIENCE, Scene
from carriage_return.world import Level

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRUE_LINE = "It is dark here. You are likely to be eaten by a grue."


class StubPerceiver:
    """The minimal thing Scene.perceive needs of a perceiver: a perception stat,
    and the detects/cares_about contract Player now implements for real."""
    def __init__(self, perception=1.0):
        self.perception = perception

    def detects(self, per_sense_values):
        return any(self.perception > -v for v in per_sense_values.values())

    def cares_about(self, percept):
        return percept.salience >= AUTO_LOG_SALIENCE


def _small_maze(shape=(12, 20)):
    """Open path with a wall border, independent of any image on disk."""
    bt = BlockTypes()
    blocks = np.full(shape, bt.id_of('path'), dtype='int')
    blocks[0, :] = blocks[-1, :] = bt.id_of('wall')
    blocks[:, 0] = blocks[:, -1] = bt.id_of('wall')
    return Maze(blocks, bt)


@pytest.fixture
def scene():
    """A bare headless Scene. cwd is the project root because Scene() loads
    level1.png as its default maze."""
    os.chdir(PROJECT_ROOT)
    return Scene()


# -- Percept.ready / mark -----------------------------------------------------

def test_never_perceived_is_ready_at_any_time():
    p = Percept("hi", salience=0.0, visibility=0.0, cooldown=300.0)
    assert p.ready(0.0)
    assert p.ready(-1e9)
    assert p.ready(1e9)


def test_cooldown_window_opens_on_mark():
    p = Percept("hi", salience=0.0, visibility=0.0, cooldown=300.0)
    t0 = 1000.0
    p.mark(t0)
    assert not p.ready(t0)
    assert not p.ready(t0 + 300.0 - 1e-6)
    assert p.ready(t0 + 300.0)
    assert p.ready(t0 + 300.0 + 1e-6)


def test_zero_cooldown_is_immediately_ready_again():
    p = Percept("hi", salience=0.0, visibility=0.0)  # cooldown defaults to 0
    p.mark(5.0)
    assert p.ready(5.0)


# -- describe -----------------------------------------------------------------

def test_describe_returns_the_text():
    p = Percept("a plain warning", salience=0.0, visibility=0.0)
    assert p.describe() == "a plain warning"
    # the perceiver arg is accepted (and ignored for now)
    assert p.describe(perceiver=StubPerceiver()) == "a plain warning"


def test_continuous_percept_tracks_value_and_describes():
    p = ContinuousPercept("hungry", salience=0.0, visibility=0.0)
    assert p.value is None
    p.update(0.5)
    assert p.value == 0.5
    assert p.describe() == "hungry"          # value-independent for now


# -- RandomDist ---------------------------------------------------------------

def test_randomdist_constant():
    d = RandomDist('constant', value=7.0)
    assert d() == 7.0
    assert list(d.sample(3)) == [7.0, 7.0, 7.0]


def test_randomdist_samples_scalars_all_types():
    for dist_type, kw in [('lognorm', dict(mean=1, stdev=0.1)),
                          ('normal', dict(mean=0, stdev=1)),
                          ('uniform', dict(low=0, high=1)),
                          ('summed_dieroll', dict(n=3, sides=6))]:
        v = RandomDist(dist_type, **kw)()
        assert np.ndim(v) == 0, f"{dist_type} gave non-scalar {v!r}"


def test_randomdist_validation():
    with pytest.raises(ValueError):
        RandomDist('nonesuch', x=1)          # unknown type
    with pytest.raises(ValueError):
        RandomDist('lognorm', mean=1)        # missing stdev
    with pytest.raises(ValueError):
        RandomDist('lognorm', mean=1, stdev=0.1, extra=9)  # unexpected param


def test_randomdist_update_params_and_switch_type():
    d = RandomDist('lognorm', mean=1, stdev=0.1)
    d.update(mean=2)                         # partial update, same type
    assert d.dist_type == 'lognorm' and d.params == {'mean': 2, 'stdev': 0.1}
    d.update(dist_type='constant', value=3.0)  # switch type + full new params
    assert d.dist_type == 'constant'
    assert d() == 3.0
    with pytest.raises(ValueError):
        d.update(stdev=0.5)                  # 'constant' has no stdev


# -- Scene.perceive: auto-log, held-quiet, cooldown, visibility gate ----------

def test_salient_percept_auto_logs_and_fires_observable(scene):
    who = StubPerceiver(perception=1.0)
    # salience >= AUTO_LOG_SALIENCE -> logged; huge visibility -> always noticed
    p = TransientPercept("you feel a chill", salience=AUTO_LOG_SALIENCE,
                         visibility=1e9)
    seen = []
    p.perceived.connect(lambda **kw: seen.append(kw))

    n0 = len(scene.log.lines)
    assert scene.perceive(p, who) is True
    assert scene.log.lines[n0:] == ["you feel a chill"]
    # perceived fired once, carrying who = the perceiver (and source = percept)
    assert len(seen) == 1
    assert seen[0]["who"] is who
    assert seen[0]["source"] is p


def test_negative_salience_percept_is_noticed_but_not_logged(scene):
    who = StubPerceiver(perception=1.0)
    p = Percept("a faint detail", salience=-1.0, visibility=1e9)

    n0 = len(scene.log.lines)
    # noticed (returns True, cooldown opened) but too quiet to speak
    assert scene.perceive(p, who) is True
    assert scene.log.lines[n0:] == []
    assert p.last_perceived is not None


def test_cooldown_suppresses_repeat_within_window(scene):
    who = StubPerceiver(perception=1.0)
    now = [1000.0]
    scene.clock = lambda: now[0]
    p = TransientPercept("it is dark here", salience=0.0, visibility=1e9,
                         cooldown=300.0)

    n0 = len(scene.log.lines)
    assert scene.perceive(p, who) is True
    assert scene.log.lines[n0:] == ["it is dark here"]

    # within the cooldown window: not noticed again, nothing re-logged
    now[0] = 1000.0 + 299.0
    assert scene.perceive(p, who) is False
    assert scene.log.lines[n0:] == ["it is dark here"]

    # once the window elapses it fires again
    now[0] = 1000.0 + 300.0
    assert scene.perceive(p, who) is True
    assert scene.log.lines[n0:] == ["it is dark here", "it is dark here"]


def test_too_faint_to_notice_is_not_marked(scene):
    who = StubPerceiver(perception=1.0)
    # visibility far below -perception: who.perception <= -visibility -> a miss
    p = TransientPercept("imperceptible", salience=0.0, visibility=-1e9)

    n0 = len(scene.log.lines)
    assert scene.perceive(p, who) is False
    assert scene.log.lines[n0:] == []
    assert p.last_perceived is None  # a miss does not open the cooldown


def test_perceive_with_real_player(scene):
    player = Player(scene)               # Player.__init__ sets scene.player = self
    # pin the acuity distribution to a constant for a deterministic roll
    player.base_perception['vision'].update(dist_type='constant', value=5.0)
    assert scene.player is player
    p = TransientPercept("you notice something", salience=0.0, visibility=1.0,
                         sense='vision')

    n0 = len(scene.log.lines)
    assert scene.perceive(p, player) is True
    assert scene.log.lines[n0:] == ["you notice something"]


def test_perceive_without_who_is_not_implemented(scene):
    # who defaults to None, meaning "anyone nearby" -- not yet implemented, and
    # explicitly NOT a shortcut for the player.
    p = TransientPercept("a noise", salience=0.0, visibility=1e9)
    with pytest.raises(NotImplementedError):
        scene.perceive(p)


# -- Level darkness -----------------------------------------------------------

def test_is_dark_at_is_none_before_compositing():
    """A level with no composited illuminance cannot judge darkness: is_dark_at
    returns None (falsy), so a grue check reading it holds its tongue."""
    level = Level('dark-test', _small_maze())
    assert level.illuminance is None
    assert level.is_dark_at((5, 5), 0.05) is None


def test_is_dark_at_true_when_composited_and_unlit():
    level = Level('dark-test', _small_maze())
    level.illuminance = np.zeros(level.field_shape, dtype='float32')
    assert level.is_dark_at((5, 5), 0.05) is True


def test_is_dark_at_false_when_composited_and_bright():
    level = Level('dark-test', _small_maze())
    level.illuminance = np.full(level.field_shape, 5e4, dtype='float32')
    assert level.is_dark_at((5, 5), 0.05) is False


def test_illuminance_at_none_without_composite_then_samples():
    level = Level('dark-test', _small_maze())
    assert level.illuminance_at((5, 6)) is None
    ss = level.supersample
    x, y = 5, 6
    level.illuminance = np.zeros(level.field_shape, dtype='float32')
    level.illuminance[y * ss, x * ss] = (1.0, 2.0, 3.0)
    assert tuple(level.illuminance_at((x, y))) == (1.0, 2.0, 3.0)


def test_invalidate_sight_flags_dirty_without_dropping_the_field():
    """A move flags illuminance dirty but leaves the last composite readable."""
    level = Level('dark-test', _small_maze())
    marker = np.full(level.field_shape, 42.0, dtype='float32')
    level.illuminance = marker
    level._illuminance_dirty = False

    level.invalidate_sight()
    assert level._illuminance_dirty is True
    assert level.illuminance is marker         # not nulled -- still readable
    assert level.illuminance_at((3, 3))[0] == 42.0


def test_illuminance_map_recomposites_when_dirty():
    """illuminance_map rebuilds a dirty (or never-built) field. A level with no
    lights composites to zeros -- no shadow provider needed."""
    level = Level('dark-test', _small_maze())
    assert level.illuminance is None
    m = level.illuminance_map()                # first build
    assert m is level.illuminance
    assert not m.any() and level._illuminance_dirty is False

    level.illuminance[:] = 99.0                # dirty it by hand + flag
    level._illuminance_dirty = True
    m2 = level.illuminance_map()               # recomposited to zeros
    assert not m2.any() and level._illuminance_dirty is False


# -- Grue integration ---------------------------------------------------------

@pytest.fixture
def grue_world():
    """A full world with a player, ready to drive the sewer's grue.

    Mirrors game.new_game's ordering (player first, then build_world) so the
    sewer wires check_darkness onto the player's movement. Returns
    (scene, world, player, sewer, now) where ``now`` is a mutable [t] holder
    backing scene.clock. The player's perception is pinned to a large constant
    so the noticing roll cannot fail.
    """
    os.chdir(PROJECT_ROOT)
    scene = Scene()
    now = [1000.0]
    scene.clock = lambda: now[0]
    player = Player(scene)
    player.base_perception['vision'] = RandomDist('constant', value=1e6)  # noticing is certain
    world = build_world(scene)
    sewer = world.levels['sewer']
    return scene, world, player, sewer, now


def _make_dark(sewer):
    """Give the sewer a composited, fully-unlit illuminance field.

    A move only flags the field dirty (invalidate_sight no longer nulls it), so
    this survives across moves -- a later dark step still reads it -- until the
    draw thread recomposites, which never runs in these headless tests.
    """
    sewer.illuminance = np.zeros(sewer.field_shape, dtype='float32')


def _grue_lines(scene):
    return [l for l in scene.log.lines if l == GRUE_LINE]


def test_stepping_into_dark_sewer_cell_warns_of_grue(grue_world):
    scene, world, player, sewer, now = grue_world
    _make_dark(sewer)
    # a real move fires location.global_changed -> the connected check_darkness
    player.location.update(sewer.maze, sewer.locations['hole'])
    assert _grue_lines(scene) == [GRUE_LINE]
    # the move flagged illuminance dirty but left the field in place
    assert sewer.illuminance is not None
    assert sewer._illuminance_dirty is True


def test_second_dark_step_within_cooldown_does_not_rewarn(grue_world):
    scene, world, player, sewer, now = grue_world
    _make_dark(sewer)
    player.location.update(sewer.maze, sewer.locations['hole'])
    assert _grue_lines(scene) == [GRUE_LINE]

    # a second dark step (field persists), clock not advanced: cooldown suppresses
    player.location.update(sewer.maze, sewer.locations['stairs_down'])
    assert _grue_lines(scene) == [GRUE_LINE]


def test_grue_rewarns_after_cooldown_elapses(grue_world):
    scene, world, player, sewer, now = grue_world
    _make_dark(sewer)
    player.location.update(sewer.maze, sewer.locations['hole'])
    assert _grue_lines(scene) == [GRUE_LINE]

    now[0] += 300.0                       # past the grue's 300 s cooldown; field still dark
    player.location.update(sewer.maze, sewer.locations['stairs_down'])
    assert _grue_lines(scene) == [GRUE_LINE, GRUE_LINE]


def test_no_grue_when_the_cell_is_lit(grue_world):
    scene, world, player, sewer, now = grue_world
    sewer.illuminance = np.full(sewer.field_shape, 5e4, dtype='float32')  # bright
    player.location.update(sewer.maze, sewer.locations['hole'])
    assert _grue_lines(scene) == []


# -- Sword integration --------------------------------------------------------

GLINT_LINE = "Something catches your eye."
SWORD_LINE = "There is a rusty sword here."


@pytest.fixture
def sword_world():
    """A full world with a player and a dungeon master, ready to walk onto the
    sewer's sword. Mirrors game.new_game's ordering (player before world) and
    makes the sewer current so dm.move_player operates on it. Perception is
    pinned high so the sword is always noticed. Returns
    (scene, player, sewer, dm, sword_cell, now)."""
    os.chdir(PROJECT_ROOT)
    scene = Scene()
    now = [1000.0]
    scene.clock = lambda: now[0]
    player = Player(scene)
    player.base_perception['vision'] = RandomDist('constant', value=1e6)
    world = build_world(scene)
    sewer = world.levels['sewer']
    scene.set_level(sewer)
    assert any(isinstance(e, Sword) for e in sewer.maze.inventory[sewer.locations['sword']])
    return scene, player, sewer, DungeonMaster(scene), sewer.locations['sword'], now


def _walk_onto_sword(player, sewer, dm, sword_cell):
    """Put the player next to the sword, then step onto its cell -- a real move
    so dm.move_player asks the sword (on_walked_on) what happens."""
    player.location.update(sewer.maze, (sword_cell[0], sword_cell[1] + 1))
    dm.move_player(player, sword_cell)


def test_sword_is_silent_in_complete_darkness(sword_world):
    scene, player, sewer, dm, sword_cell, now = sword_world
    sewer.illuminance = np.zeros(sewer.field_shape, dtype='float32')  # true darkness
    _walk_onto_sword(player, sewer, dm, sword_cell)
    # neither of the sword's own percepts fires in true darkness -- unrelated
    # to whether the sewer's grue warning (a separate, generic-sense percept
    # tied to level darkness, not the sword) also fires on the same step.
    assert GLINT_LINE not in scene.log.lines
    assert SWORD_LINE not in scene.log.lines


def test_sword_is_plain_when_the_cell_is_lit(sword_world):
    scene, player, sewer, dm, sword_cell, now = sword_world
    sewer.illuminance = np.full(sewer.field_shape, 5e4, dtype='float32')  # bright
    _walk_onto_sword(player, sewer, dm, sword_cell)
    assert SWORD_LINE in scene.log.lines
    # GLINT_LINE may also fire alongside it -- percepts are independent now


# -- VisualPercept.detectability -----------------------------------------------

class _FakeLevel:
    """The minimal thing VisualPercept.detectability needs from a Level."""
    def __init__(self, luminance):
        self._luminance = luminance

    def luminance_at(self, pos):
        return self._luminance


def _fake_positioned(slot, level):
    """A minimal stand-in for anything with `.location.global_location.slot`
    and `.location.global_location.container.level` -- enough for both the
    percept's owning entity and the perceiver."""
    loc = SimpleNamespace(slot=slot, container=SimpleNamespace(level=level))
    return SimpleNamespace(location=SimpleNamespace(global_location=loc))


def test_visual_percept_is_negative_infinity_in_true_darkness():
    level = _FakeLevel(luminance=0.0)
    entity = _fake_positioned((5, 5), level)
    perceiver = _fake_positioned((5, 5), level)
    p = VisualPercept("x", salience=0.0, visibility=1.0, entity=entity)
    assert p.detectability(perceiver) == {'vision': float('-inf')}


def test_visual_percept_saturates_at_visibility_in_bright_light():
    level = _FakeLevel(luminance=1e5)  # far above DIFFUSE_SCALE (2.0 lux)
    entity = _fake_positioned((5, 5), level)
    perceiver = _fake_positioned((5, 5), level)  # distance 0
    p = VisualPercept("x", salience=0.0, visibility=2.5, entity=entity)
    assert p.detectability(perceiver)['vision'] == pytest.approx(2.5)


def test_visual_percept_glint_clears_before_identify_in_dim_light():
    level = _FakeLevel(luminance=0.1)  # dim, well below DIFFUSE_SCALE
    entity = _fake_positioned((5, 5), level)
    perceiver = _fake_positioned((5, 5), level)
    identify = VisualPercept("id", salience=0.0, visibility=0.0, entity=entity)
    glint = VisualPercept("glint", salience=0.0, visibility=6.0, entity=entity)
    identify_vis = identify.detectability(perceiver)['vision']
    glint_vis = glint.detectability(perceiver)['vision']
    # a realistic acuity (~1.0, the game's default lognorm mean) clears the
    # glint's bar but not the identify bar -- this is what makes "catches
    # your eye" trigger before "there is a rusty sword here" as light rises.
    assert glint_vis > -1.0
    assert identify_vis < -1.0


def test_visual_percept_attenuates_with_distance():
    level = _FakeLevel(luminance=1e5)
    entity = _fake_positioned((0, 0), level)
    near = _fake_positioned((1, 0), level)
    far = _fake_positioned((10, 0), level)
    p = VisualPercept("x", salience=0.0, visibility=6.0, entity=entity)
    near_vis = p.detectability(near)['vision']
    far_vis = p.detectability(far)['vision']
    assert far_vis < near_vis
