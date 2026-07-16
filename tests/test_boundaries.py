"""Enforce the game/renderer module boundary.

Game-side modules must be importable without pulling in any rendering or GUI
library. The vispy side of the codebase is ui.py, input.py, interpreter.py,
graphics.py, and render_vispy.py; everything else is game state and must stay
renderer-agnostic (see ARCHITECTURE.md).
"""
import subprocess
import sys

GAME_MODULES = [
    'array_cache',
    'blocktypes',
    'command',
    'dialogue',
    'dm',
    'entity',
    'errors',
    'events',
    'geometry',
    'inventory',
    'item',
    'layers',
    'location',
    'maze',
    'monster',
    'player',
    'scene',
    'sprite',
]

FORBIDDEN = ['vispy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'OpenGL']

CHECK_SCRIPT = """
import sys
for mod in %r:
    __import__('carriage_return.' + mod)
bad = sorted(m for m in sys.modules
             if any(m == f or m.startswith(f + '.') for f in %r))
if bad:
    print('forbidden modules imported:', ', '.join(bad))
    sys.exit(1)
print('OK')
""" % (GAME_MODULES, FORBIDDEN)


def test_game_modules_import_no_rendering_libs():
    result = subprocess.run([sys.executable, '-c', CHECK_SCRIPT],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OK' in result.stdout
