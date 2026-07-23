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

import numpy as np
import pytest

from carriage_return.blocktypes import BlockTypes
from carriage_return.levels import build_world
from carriage_return.maze import Maze
from carriage_return.perception import ContinuousPercept, Percept, TransientPercept
from carriage_return.player import Player
from carriage_return.random import RandomDist
from carriage_return.scene import AUTO_LOG_SALIENCE, Scene
from carriage_return.world import Level

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRUE_LINE = "It is dark here. You are likely to be eaten by a grue."


class StubPerceiver:
    """The minimal thing Scene.perceive needs of a perceiver: a perception stat."""
    def __init__(self, perception=1.0):
        self.perception = perception


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
    player.base_perception.update(dist_type='constant', value=5.0)
    assert scene.player is player
    p = TransientPercept("you notice something", salience=0.0, visibility=1.0)

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
    player.base_perception = RandomDist('constant', value=1e6)  # noticing is certain
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
