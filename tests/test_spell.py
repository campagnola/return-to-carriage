"""Spell casting: the fireball/lightning mobs, the cast prompt, and the
interpreter wiring.

Everything here is headless and synchronous. The spells are built with
``start=False`` so no animation thread is spawned; the fireball is driven a
tick at a time through ``advance(dt)`` (like the movement handler's injectable
clock), and lighting is a one-shot that we tear down explicitly. The cast
prompt runs through its real DialogSession loop, fed synthetic key events.
"""
import os

import pytest

from carriage_return.blocktypes import BlockTypes
from carriage_return.dialogs import CastPrompt, DialogSession, open_cast, run_cast
from carriage_return.input import InputDispatcher, KeyPress
from carriage_return.interpreter import CommandInterpreter
from carriage_return.maze import Maze
from carriage_return.player import Player
from carriage_return.scene import Scene
from carriage_return import spell
from carriage_return.world import Level

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOIN_TIMEOUT = 10.0


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)
    return Scene()


def corridor(rows=5, cols=12, path_row=2):
    """A one-cell-high open corridor walled on every side, as its own Level.

    Path runs along *path_row* from x=1..cols-2; everything else is wall, so a
    bolt fired rightward from x=1 stops when it reaches the wall at x=cols-1.
    """
    bt = BlockTypes()
    maze = Maze.filled((rows, cols), bt, 'wall')
    maze.blocks[path_row, 1:cols - 1] = bt.id_of('path')
    Level('corridor', maze)   # sets maze.level
    return maze


def open_room(size=20):
    bt = BlockTypes()
    maze = Maze.filled((size, size), bt, 'path')
    maze.blocks[0, :] = maze.blocks[-1, :] = bt.id_of('wall')
    maze.blocks[:, 0] = maze.blocks[:, -1] = bt.id_of('wall')
    Level('room', maze)
    return maze


# -- direction glyphs -------------------------------------------------------

def test_lightning_glyphs():
    g = spell.Lightning._glyph
    assert g((1, 0)) == '-' and g((-1, 0)) == '-'
    assert g((0, 1)) == '|' and g((0, -1)) == '|'
    assert g((1, 1)) == '/' and g((-1, -1)) == '/'      # rises to the right
    assert g((1, -1)) == '\\' and g((-1, 1)) == '\\'    # falls to the right


# -- fireball ---------------------------------------------------------------

def test_fireball_places_light_and_sprite(scene):
    maze = corridor()
    actors = scene.sprite_layers['actors']
    n0 = len(actors)
    fb = spell.Fireball(scene, maze, pos=(1, 2), direction=(1, 0), start=False)

    assert fb.lights and fb.lights[0] in maze.level.lights
    assert len(actors) == n0 + 1
    # sprite sits on the start cell
    assert tuple(fb.sprite.position[0][:2]) == (1.0, 2.0)


def test_fireball_steps_one_cell_at_a_time(scene):
    maze = corridor()
    fb = spell.Fireball(scene, maze, pos=(1, 2), direction=(1, 0), start=False)
    # SPEED*dt = 1.4 cells -> truncates to a single cell step
    assert fb.advance(dt=1.0 / spell.Fireball.SPEED * 1.4) is True
    assert tuple(fb.pos) == (2, 2)


def test_fireball_travels_and_snuffs_at_wall(scene):
    maze = corridor(cols=12)          # path x=1..10, wall at x=11
    actors = scene.sprite_layers['actors']
    n0 = len(actors)
    fb = spell.Fireball(scene, maze, pos=(1, 2), direction=(1, 0), start=False)

    # one big tick carries it the length of the corridor; it must stop at the
    # wall, not tunnel through it
    alive = fb.advance(dt=10.0)
    assert alive is False
    assert fb._done
    assert tuple(fb.pos) == (10, 2)               # last open cell reached
    assert len(actors) == n0                       # sprite removed

    # the fireball's own light is gone, but it leaves a glowing hot spot on the
    # wall it struck (x=11): one light, belonging to that Heat mob
    assert fb.heat is not None
    assert fb.heat.pos == (11, 2)
    assert maze.level.lights == [fb.heat.light]


def test_fireball_destroy_is_idempotent(scene):
    maze = corridor()
    fb = spell.Fireball(scene, maze, pos=(1, 2), direction=(1, 0), start=False)
    fb.destroy()
    fb.destroy()   # must not raise (e.g. removing an already-freed sprite/light)
    assert fb._done


# -- lightning --------------------------------------------------------------

def test_lightning_traces_full_length_in_open_room(scene):
    maze = open_room(30)
    actors = scene.sprite_layers['actors']
    n0 = len(actors)
    bolt = spell.Lightning(scene, maze, pos=(5, 5), direction=(1, 0), start=False)

    assert len(bolt.cells) == spell.Lightning.LENGTH
    assert bolt.cells[0] == (5, 5)
    assert len(bolt.lights) == spell.Lightning.LENGTH
    assert len(actors) == n0 + spell.Lightning.LENGTH
    for light in bolt.lights:
        assert light in maze.level.lights


def test_lightning_stops_at_wall(scene):
    # a 4-wide interior means the rightward bolt can only reach a few cells
    maze = corridor(rows=5, cols=6, path_row=2)   # path x=1..4
    bolt = spell.Lightning(scene, maze, pos=(1, 2), direction=(1, 0), start=False)
    assert len(bolt.cells) <= 4
    assert all(maze.blocktype_at(y, x)['walkable'] for (x, y) in bolt.cells)


def test_lightning_flash_toggles_lights_and_sprites(scene):
    import numpy as np
    maze = open_room(30)
    bolt = spell.Lightning(scene, maze, pos=(5, 5), direction=(1, 0), start=False)

    bolt._set_lit(False)
    assert all(light.brightness == 0.0 for light in bolt.lights)
    assert np.isnan(bolt.sprite.position).all()

    bolt._set_lit(True)
    assert all(light.brightness == 1.0 for light in bolt.lights)
    assert not np.isnan(bolt.sprite.position).any()


def test_lightning_teardown_removes_lights_and_sprites(scene):
    maze = open_room(30)
    actors = scene.sprite_layers['actors']
    n0 = len(actors)
    bolt = spell.Lightning(scene, maze, pos=(5, 5), direction=(0, 1), start=False)
    bolt.destroy()
    assert maze.level.lights == []
    assert len(actors) == n0


# -- cast prompt model ------------------------------------------------------

def test_cast_prompt_resolves_by_substring():
    p = CastPrompt(spell.SPELLS)
    p.text = "bal"
    assert p.resolve() == "fireball"
    assert p.phase == "direction" and p.spell_name == "fireball"


def test_cast_prompt_prefix_match():
    p = CastPrompt(spell.SPELLS)
    p.text = "lit"
    assert p.resolve() == "lightning"


def test_cast_prompt_unknown_name_fizzles():
    p = CastPrompt(spell.SPELLS)
    p.text = "xyzzy"
    assert p.resolve() is None
    assert p.phase == "name"
    assert p.message


def test_cast_prompt_ambiguous_reports():
    p = CastPrompt(spell.SPELLS)
    p.text = "l"                       # in both firebaLl and Lightning
    assert p.resolve() is None
    assert "mean" in p.message.lower()


# -- cast prompt key loop ---------------------------------------------------

def _type(session, text):
    for ch in text:
        session.post(KeyPress(ch, text=ch))


def test_run_cast_type_name_then_direction():
    p = CastPrompt(spell.SPELLS)
    session = DialogSession(lambda s: run_cast(s, p)).start()
    _type(session, "bal")
    session.post(KeyPress('Enter'))
    session.post(KeyPress('Right'))
    session.join(timeout=JOIN_TIMEOUT)
    assert session.result == ("fireball", "Right")


def test_run_cast_escape_cancels():
    p = CastPrompt(spell.SPELLS)
    session = DialogSession(lambda s: run_cast(s, p)).start()
    session.post(KeyPress('Escape'))
    session.join(timeout=JOIN_TIMEOUT)
    assert session.result is None


def test_run_cast_bad_name_stays_open_then_recovers():
    p = CastPrompt(spell.SPELLS)
    session = DialogSession(lambda s: run_cast(s, p)).start()
    _type(session, "zzz")
    session.post(KeyPress('Enter'))        # fizzles, prompt stays up
    for ch in "zzz":                        # clear the bad text
        session.post(KeyPress('Backspace'))
    _type(session, "lit")
    session.post(KeyPress('Enter'))
    session.post(KeyPress('Up'))
    session.join(timeout=JOIN_TIMEOUT)
    assert session.result == ("lightning", "Up")


# -- interpreter wiring -----------------------------------------------------

@pytest.fixture
def dispatcher():
    InputDispatcher.reset()
    disp = InputDispatcher()
    yield disp
    InputDispatcher.reset()


def test_interpreter_cast_builds_spell_at_player(scene, dispatcher, monkeypatch):
    player = Player(scene)
    player.location.update(scene.maze, [7, 7])
    interp = CommandInterpreter(scene)

    built = []
    monkeypatch.setitem(spell.SPELLS, 'fireball',
                        lambda *a, **k: built.append(a) or object())

    session = interp.cast([])                    # opens the modal prompt
    for ch in "bal":
        dispatcher.dispatch(KeyPress(ch, text=ch))
    dispatcher.dispatch(KeyPress('Enter'))
    dispatcher.dispatch(KeyPress('Left'))
    session.join(timeout=JOIN_TIMEOUT)            # _cast runs on this thread

    assert len(built) == 1
    bscene, bmaze, bpos, bdir = built[0]
    assert bscene is scene
    assert bmaze is scene.maze
    assert list(bpos) == [7, 7]
    assert bdir == spell.DIRECTIONS['Left']
