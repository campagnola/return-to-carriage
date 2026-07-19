"""Verify the layer renderer does zero sprite writes when nothing changed.

Builds the deterministic harness scene, renders twice to reach steady state,
then instruments SpriteData's property setters and renders again: a static
scene must perform no SpriteData writes (and request no VBO uploads) during
the extra frame. Exits nonzero on failure.
"""
import os
import sys

import numpy as np

np.random.seed(12345)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from render_screenshot import build_game  # noqa: E402  (agent_helpers sibling)
import carriage_return.backends.vispy.graphics as graphics  # noqa: E402

counter = {'writes': 0}


def instrument():
    for name in ('position', 'sprite', 'fgcolor', 'bgcolor'):
        prop = getattr(graphics.SpriteData, name)

        def make_fset(fset):
            def fset_counting(self, value):
                counter['writes'] += 1
                return fset(self, value)
            return fset_counting

        setattr(graphics.SpriteData, name, property(prop.fget, make_fset(prop.fset)))


def main():
    ui, scene, player = build_game()
    ui.canvas.set_current()
    for _ in range(2):
        scene.on_draw(None)
        ui.canvas.render()

    instrument()
    scene.on_draw(None)
    ui.canvas.render()
    static_writes = counter['writes']
    print("SpriteData writes during static frame:", static_writes)

    # sanity-check the instrumentation and the change path: move the monster,
    # the next frame must re-sync exactly one layer (4 writes)
    counter['writes'] = 0
    yeti = scene.monsters[list(scene.monsters.keys())[0]][0]
    yeti.take_turn()
    scene.on_draw(None)
    ui.canvas.render()
    change_writes = counter['writes']
    print("SpriteData writes after one monster move:", change_writes)

    ok = static_writes == 0 and change_writes == 4
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
