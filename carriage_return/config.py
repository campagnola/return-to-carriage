"""Process-wide runtime configuration, set once at startup.

:data:`test_mode` is a single boolean flipped on by the ``--test`` CLI flag
(see ``return_to_carriage.py``) before the game is built. It exists to shorten
slow, real-time behaviours so a human testing the game by hand does not have to
sit through them -- the first being the eye-adaptation time constants (see
:mod:`.adaptation`), which otherwise take ~20 s to open the eye up in the dark.

It is off by default, so the real game and the screenshot harness behave
identically unless the flag is passed. Read it as an attribute --
``config.test_mode`` -- never ``from .config import test_mode``, which would
snapshot the value at import time and miss the flag being set at startup.

Game-side module: pure Python only, no rendering library (see
tests/test_boundaries.py).
"""

#: Process-wide test mode. Set True by the ``--test`` CLI flag before the world
#: is built; read wherever a slow, real-time behaviour should be sped up for
#: hands-on testing. Always accessed as ``config.test_mode`` so consumers see
#: the value set at startup, not a stale import-time copy.
test_mode = False
