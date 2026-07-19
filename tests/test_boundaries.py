"""Enforce the game/renderer module boundary.

One rule: no module in ``carriage_return/`` outside ``carriage_return/backends/``
may import vispy, Qt, or OpenGL. The game-side module list is discovered by
walking the package tree (not hand-maintained), so new modules — including
everything added to the dialogs subpackage — are covered automatically. Each
game-side module is imported in a subprocess and ``sys.modules`` is checked
for forbidden libraries afterward (see ARCHITECTURE.md).
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, 'carriage_return')

FORBIDDEN = ['vispy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'OpenGL']


def discover_game_modules():
    """Dotted names of every module under carriage_return/ except backends/.

    Walks the filesystem (not pkgutil) so it never has to import anything —
    in particular it never has to import carriage_return.backends, which is
    exactly the package this test must not trust.
    """
    modules = []
    for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
        rel = os.path.relpath(dirpath, PACKAGE_ROOT)
        parts = [] if rel == '.' else rel.split(os.sep)

        # don't descend into backends/ or caches at all
        dirnames[:] = [d for d in dirnames
                       if d not in ('backends', '__pycache__') and not d.startswith('.')]

        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            stem = filename[:-3]
            if stem == '__init__':
                mod_parts = parts
            else:
                mod_parts = parts + [stem]
            if not mod_parts:
                continue  # skip the top-level carriage_return/__init__.py itself
            modules.append('.'.join(mod_parts))
    return sorted(set(modules))


GAME_MODULES = discover_game_modules()

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


def test_discovery_found_the_game_modules():
    # sanity check on the discovery walk itself, so a broken walk (e.g. one
    # that silently finds nothing) fails loudly instead of vacuously passing
    # test_game_modules_import_no_rendering_libs below
    assert len(GAME_MODULES) >= 20
    for expected in ('scene', 'layers', 'input', 'interpreter', 'hud',
                     'dialogs', 'dialogs.menu', 'dialogs.pager', 'dialogs.session'):
        assert expected in GAME_MODULES
    assert not any(m == 'backends' or m.startswith('backends.') for m in GAME_MODULES)


def test_game_modules_import_no_rendering_libs():
    result = subprocess.run([sys.executable, '-c', CHECK_SCRIPT],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OK' in result.stdout
