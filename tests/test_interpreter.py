"""Typed console commands driving the game-side action layer.

The interpreter shares its take/read/drop triage with the t/r/d shortcuts:
an ambiguous typed 'take'/'drop' opens the same modal Menu the shortcuts do
(a DialogSession on the InputDispatcher stack, a grid in scene.grids), driven
here with synthetic InputEvents. The old console letter-menu is gone.
"""
import os

import numpy as np
import pytest

from carriage_return.dialogs import DialogSession
from carriage_return.input import InputDispatcher, KeyPress
from carriage_return.interpreter import CommandInterpreter
from carriage_return.item import Scroll, Torch
from carriage_return.player import Player
from carriage_return.scene import Scene

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeVisibility:
    def __init__(self, scene):
        self.shape = scene.field_shape[:2] + (4,)

    def render(self, pos, read=True):
        return np.full(self.shape, 255, dtype='ubyte')


@pytest.fixture
def dispatcher():
    InputDispatcher.reset()
    disp = InputDispatcher()
    yield disp
    InputDispatcher.reset()


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)
    scene = Scene()
    scene.visibility = FakeVisibility(scene)
    player = Player(scene)
    player.location.update(scene.maze, [7, 7])
    return scene


@pytest.fixture
def interp(scene):
    return CommandInterpreter(scene)


def press(session, *keys):
    for key in keys:
        session.post(KeyPress(key))


def finish(session):
    session.join(10)
    assert session.finished.is_set() and session.error is None


def test_typed_take_ambiguous_opens_menu(scene, interp, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    torch = Torch(location=(scene.maze, pos), scene=scene)

    interp('take')  # two items => modal menu
    assert any(line == "> take" for line in scene.log.lines)  # command echoed
    session = dispatcher.handlers[-1]
    assert isinstance(session, DialogSession)
    assert len(scene.grids) == 1

    press(session, 'Down', 'Space', 'Enter')  # take the torch
    finish(session)

    assert torch.location.container is scene.player
    assert scroll.location.container is scene.maze
    assert "Taken: %s." % torch.description in scene.log.lines
    assert len(scene.grids) == 0 and dispatcher.handlers == []


def test_typed_take_single_skips_menu(scene, interp, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    interp('take')
    assert scroll.location.container is scene.player
    assert len(scene.grids) == 0


def test_typed_take_by_name(scene, interp, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    Torch(location=(scene.maze, pos), scene=scene)
    interp('take scroll')  # unambiguous name: no menu
    assert scroll.location.container is scene.player
    assert len(scene.grids) == 0


def test_typed_take_nothing(scene, interp, dispatcher):
    interp('take')
    assert "You take, but nothing gives." in scene.log.lines


def test_typed_drop_ambiguous_opens_menu(scene, interp, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    torch = Torch(location=(scene.maze, pos), scene=scene)
    scene.player.take(scroll)
    scene.player.take(torch)

    interp('drop')  # two held items => modal menu
    session = dispatcher.handlers[-1]
    assert isinstance(session, DialogSession)

    press(session, 'Space', 'Enter')  # drop the first inventory item
    finish(session)

    dropped = [i for i in (scroll, torch) if i.location.container is scene.maze]
    assert len(dropped) == 1
    assert len(scene.grids) == 0 and dispatcher.handlers == []


def test_typed_drop_by_name(scene, interp, dispatcher):
    pos = scene.player.location.slot
    torch = Torch(location=(scene.maze, pos), scene=scene)
    scene.player.take(torch)
    interp('drop torch')
    assert torch.location.container is scene.maze
    assert len(scene.grids) == 0


def test_unknown_verb(scene, interp, dispatcher):
    interp('dance')
    assert scene.log.lines[-1] == 'You lost me at "dance"'


def test_empty_command(scene, interp, dispatcher):
    interp('')
    assert scene.log.lines[-1] == 'You lost me at ""'
