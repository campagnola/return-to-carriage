"""Take/read/drop player actions driven through the real dialog pipeline.

The ambiguous paths run the genuine threaded dialog pipeline: the interpreter
calls ``dialogs.open_menu`` / ``open_pager``, which push a DialogSession onto
the InputDispatcher stack and a grid into ``scene.grids``. Tests find the
session as ``dispatcher.handlers[-1]``, drive it with synthetic InputEvents,
and assert on inventory, the map, ``scene.grids``, and ``scene.log`` — no
rendering library is involved.
"""
import os

import numpy as np
import pytest

from carriage_return.dialogs import DialogSession
from carriage_return.errors import ActionError
from carriage_return.input import InputDispatcher, KeyPress
from carriage_return.interpreter import CommandInterpreter
from carriage_return.item import Item, Scroll, Torch
from carriage_return.player import Player
from carriage_return.scene import Scene

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeVisibility:
    """Numpy-only visibility provider: everything fully visible."""
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
    os.chdir(PROJECT_ROOT)  # level1.png is loaded from cwd
    scene = Scene()
    scene.visibility = FakeVisibility(scene)
    player = Player(scene)
    player.location.update(scene.maze, [7, 7])
    return scene


@pytest.fixture
def interp(scene):
    return CommandInterpreter(scene)


@pytest.fixture
def messages(scene):
    # scene.log.lines, empty at fixture time, so tests read exactly the
    # messages written during the test
    assert scene.log.lines == []
    return scene.log.lines


def press(session, *keys):
    for key in keys:
        session.post(KeyPress(key))


def finish(session):
    session.join(10)
    assert session.finished.is_set() and session.error is None


def grid_text(grid):
    return '\n'.join(''.join(grid.registry.chars[i] for i in grid.glyph[r])
                     for r in range(grid.shape[0]))


def only_grid_text(scene):
    grids = list(scene.grids)
    assert len(grids) == 1
    return grid_text(grids[0])


# -- direct player mechanics (no dialog) -------------------------------------

def test_items_at_excludes_non_items(scene):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    assert scene.items_at(pos) == [scroll]  # the player shares the slot
    assert scene.items_at((0, 0)) == []


def test_player_take_and_drop(scene):
    pos = scene.player.location.slot
    player = scene.player
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    torch = Torch(location=(scene.maze, pos), scene=scene)

    slot = player.take(scroll)
    assert scroll.location.container is player
    assert scroll in player.inventory[slot]
    assert scroll not in scene.maze.inventory[pos]
    assert np.isnan(scroll.sprite.sprite.position).all()  # hidden from the map

    player.take(torch)  # second hand
    third = Torch(location=(scene.maze, pos), scene=scene)
    with pytest.raises(ActionError, match="hands are full"):
        player.take(third)

    boulder = Item(location=(scene.maze, pos), scene=scene)  # takeable = False
    with pytest.raises(ActionError):
        player.take(boulder)

    player.drop(scroll)
    assert scroll.location.container is scene.maze
    assert scroll in scene.maze.inventory[pos]
    with pytest.raises(ActionError, match="not holding"):
        player.drop(scroll)


# -- take triage -------------------------------------------------------------

def test_take_nothing_here(scene, interp, messages, dispatcher):
    interp.take([])
    assert messages == ["You take, but nothing gives."]
    assert list(scene.grids) == []  # no dialog opened


def test_take_single_item_skips_menu(scene, interp, messages, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    interp.take([])
    assert scroll.location.container is scene.player
    assert messages == ["Taken: %s." % scroll.description]
    assert list(scene.grids) == []  # single item => no menu


def test_take_menu_selection(scene, interp, messages, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    torch = Torch(location=(scene.maze, pos), scene=scene)

    interp.take([])
    session = dispatcher.handlers[-1]
    assert isinstance(session, DialogSession)
    text = only_grid_text(scene)  # the menu grid appeared, listing both items
    assert scroll.description in text and torch.description in text

    press(session, 'Down', 'Space', 'Enter')  # check the torch only
    finish(session)

    assert torch.location.container is scene.player
    assert scroll.location.container is scene.maze
    assert messages == ["Taken: %s." % torch.description]
    assert list(scene.grids) == []                 # menu grid removed
    assert dispatcher.handlers == []               # session popped


def test_take_menu_cancel(scene, interp, messages, dispatcher):
    pos = scene.player.location.slot
    Scroll(location=(scene.maze, pos), scene=scene)
    Torch(location=(scene.maze, pos), scene=scene)

    interp.take([])
    session = dispatcher.handlers[-1]
    press(session, 'Escape')
    finish(session)

    assert scene.items_at(pos) != []   # nothing was taken
    assert messages == []              # and the take callback never ran
    assert list(scene.grids) == []
    assert dispatcher.handlers == []


def test_take_by_name(scene, interp, messages, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    Torch(location=(scene.maze, pos), scene=scene)

    interp.take(['scroll'])  # unambiguous name match: no menu
    assert scroll.location.container is scene.player
    assert messages == ["Taken: %s." % scroll.description]
    assert list(scene.grids) == []


# -- read triage -------------------------------------------------------------

def test_read_nothing(scene, interp, messages, dispatcher):
    interp.read([])
    assert messages == ["You have nothing to read."]


def test_read_single_opens_pager(scene, interp, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    scene.player.take(scroll)

    interp.read([])
    session = dispatcher.handlers[-1]
    assert isinstance(session, DialogSession)
    assert scroll.description in only_grid_text(scene)

    press(session, 'Right', 'Escape')
    finish(session)

    assert scroll in scene.items          # reading no longer destroys it
    assert scroll.location.container is scene.player
    assert list(scene.grids) == []
    assert dispatcher.handlers == []


def test_read_menu_multiple(scene, interp, dispatcher):
    pos = scene.player.location.slot
    a = Scroll(location=(scene.maze, pos), scene=scene)
    b = Scroll(location=(scene.maze, pos), scene=scene)

    interp.read([])
    session = dispatcher.handlers[-1]
    assert isinstance(session, DialogSession)
    assert only_grid_text(scene).count(Scroll.name) >= 2  # both readables listed

    press(session, 'Escape')  # cancel: nothing read, no pager opens
    finish(session)
    assert a in scene.items and b in scene.items
    assert list(scene.grids) == []
    assert dispatcher.handlers == []


# -- drop triage -------------------------------------------------------------

def test_drop_nothing(scene, interp, messages, dispatcher):
    interp.drop([])
    assert messages == ["You have nothing to drop."]
    assert list(scene.grids) == []


def test_drop_menu_selection(scene, interp, dispatcher):
    pos = scene.player.location.slot
    scroll = Scroll(location=(scene.maze, pos), scene=scene)
    torch = Torch(location=(scene.maze, pos), scene=scene)
    scene.player.take(scroll)
    scene.player.take(torch)

    interp.drop([])
    session = dispatcher.handlers[-1]
    assert isinstance(session, DialogSession)
    assert scroll.description in only_grid_text(scene)

    press(session, 'Space', 'Enter')  # drop the first inventory item
    finish(session)

    dropped = [i for i in (scroll, torch) if i.location.container is scene.maze]
    assert len(dropped) == 1
    assert list(scene.grids) == []
    assert dispatcher.handlers == []
