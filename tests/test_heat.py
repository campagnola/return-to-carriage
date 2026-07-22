"""Heat mobs and the blackbody colour they glow with.

Headless and synchronous, like test_spell: the Heat is built with ``start=False``
so no cooling thread spawns, and driven a tick at a time through ``advance(dt)``.
"""
import os

import numpy as np
import pytest

from carriage_return.blocktypes import BlockTypes
from carriage_return.heat import AMBIENT_TEMP, Heat, blackbody_color
from carriage_return.maze import Maze
from carriage_return.scene import Scene
from carriage_return.world import Level

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def scene():
    os.chdir(PROJECT_ROOT)
    return Scene()


def room(size=10):
    bt = BlockTypes()
    maze = Maze.filled((size, size), bt, 'path')
    Level('room', maze)
    return maze


# -- blackbody colour -------------------------------------------------------

def test_blackbody_channels_are_normalised_0_1():
    for t in (1000, 3000, 6500, 12000, 40000):
        r, g, b = blackbody_color(t)
        assert all(0.0 <= c <= 1.0 for c in (r, g, b))
        assert max(r, g, b) == pytest.approx(1.0, abs=1e-6)   # brightest is ~1


def test_blackbody_hot_is_bluer_than_cool():
    # cooler glows are red-dominant; hotter ones pull toward blue-white
    cool = blackbody_color(1500)
    hot = blackbody_color(12000)
    assert cool[0] > cool[2]          # red beats blue when cool
    assert hot[2] >= hot[0]           # blue catches/overtakes red when hot
    # red is pinned at full across the range; blue is what climbs with heat
    assert hot[2] > cool[2]


# -- cooling ----------------------------------------------------------------

def test_heat_places_a_pinned_light(scene):
    maze = room()
    n0 = len(maze.level.lights)
    h = Heat(scene, maze, pos=(4, 4), temperature=3000.0, start=False)
    assert h.light in maze.level.lights
    assert len(maze.level.lights) == n0 + 1
    # glyph-less: it adds no actor sprite
    assert h.temperature == 3000.0


def test_heat_cools_toward_ambient_and_dims(scene):
    maze = room()
    h = Heat(scene, maze, pos=(4, 4), temperature=3000.0, start=False)
    hot_b = h.light.brightness
    assert hot_b > 0

    # a short step cools it but leaves it glowing
    assert h.advance(dt=0.5) is True
    assert AMBIENT_TEMP < h.temperature < 3000.0
    assert 0 < h.light.brightness < hot_b


def test_heat_destroys_itself_once_cold(scene):
    maze = room()
    h = Heat(scene, maze, pos=(4, 4), temperature=3000.0, start=False)
    # a long step drops it essentially to ambient
    assert h.advance(dt=100.0) is False
    assert h._done
    assert maze.level.lights == []               # light removed on destroy
    assert h.temperature - AMBIENT_TEMP <= Heat.DESTROY_MARGIN


def test_heat_color_tracks_temperature(scene):
    maze = room()
    h = Heat(scene, maze, pos=(4, 4), temperature=9000.0, start=False)
    assert h.light.color == blackbody_color(9000.0)
    h.advance(dt=1.0)
    assert h.light.color == blackbody_color(h.temperature)


def test_heat_destroy_is_idempotent(scene):
    maze = room()
    h = Heat(scene, maze, pos=(4, 4), temperature=3000.0, start=False)
    h.destroy()
    h.destroy()
    assert h._done


def test_cooling_is_exact_exponential_independent_of_step(scene):
    # one big step must equal many small ones (Newton's law, closed form)
    maze = room()
    a = Heat(scene, maze, pos=(1, 1), temperature=5000.0, start=False)
    b = Heat(scene, maze, pos=(2, 2), temperature=5000.0, start=False)
    a.advance(dt=0.9)
    for _ in range(9):
        b.advance(dt=0.1)
    assert a.temperature == pytest.approx(b.temperature, rel=1e-6)
