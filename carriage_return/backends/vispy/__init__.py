"""The vispy/Qt/OpenGL rendering backend.

Everything in this subpackage is allowed to import vispy, Qt, or OpenGL —
it is the only part of the codebase that may (see tests/test_boundaries.py).
Re-exports the pieces callers wire together (see return_to_carriage.py):
``MainWindow`` (the canvas/camera/frame-tick shell), ``VispySceneRenderer``
(sprite-layer + shadow rendering), ``GridRenderer`` (generic CharGridLayer
renderer), and ``CanvasInputSource`` (keyboard events -> InputEvents).
"""
from .window import MainWindow
from .render import VispySceneRenderer
from .grids import GridRenderer
from .input import CanvasInputSource

__all__ = ['MainWindow', 'VispySceneRenderer', 'GridRenderer', 'CanvasInputSource']
