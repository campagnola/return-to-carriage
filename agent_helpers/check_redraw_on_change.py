"""Verify that game-state changes schedule a redraw on their own (bug: key
presses were processed but nothing repainted until the mouse forced a draw).

Runs the real vispy event loop:
 phase 1 (settle): let startup draws finish
 phase 2 (quiet):  no input -> must see (almost) no draws, proving the layer
                   observers don't cause a continuous redraw loop
 phase 3 (move):   move the player via location.update (the same path a key
                   press drives) -> must see at least one draw

Exits 0 on success.
"""
import os
import sys

import numpy as np

np.random.seed(12345)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

import vispy.app

from render_screenshot import build_game

ui, scene, renderer, player = build_game()

draw_count = [0]
ui.canvas.events.draw.connect(lambda ev: draw_count.__setitem__(0, draw_count[0] + 1))

state = {'phase': 'settle', 'quiet_start': None, 'move_start': None}
failures = []


def step(ev):
    if state['phase'] == 'settle' and ev.elapsed > 1.0:
        state['phase'] = 'quiet'
        state['quiet_start'] = draw_count[0]
    elif state['phase'] == 'quiet' and ev.elapsed > 2.0:
        quiet_draws = draw_count[0] - state['quiet_start']
        print("draws during 1s quiet period: %d" % quiet_draws)
        if quiet_draws > 5:
            failures.append("continuous redraw loop: %d draws while idle" % quiet_draws)
        state['phase'] = 'move'
        state['move_start'] = draw_count[0]
        player.location.update(scene.maze, [8, 7])
    elif state['phase'] == 'move' and ev.elapsed > 3.0:
        move_draws = draw_count[0] - state['move_start']
        print("draws after player move: %d" % move_draws)
        if move_draws < 1:
            failures.append("player move did not schedule a redraw")
        timer.stop()
        vispy.app.quit()


timer = vispy.app.Timer(interval=0.05, connect=step, start=True)
vispy.app.run()

if failures:
    for f in failures:
        print("FAIL:", f)
    sys.exit(1)
print("PASS")
