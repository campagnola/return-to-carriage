import numpy as np
import vispy.scene, vispy.app
import vispy.util.ptime as ptime

from .grids import GridRenderer
from .input import CanvasInputSource


class MainWindow:
    """Implements the desktop window: canvas, OpenGL context, cameras, and
    the connection between OS event sources and the game.

    All text panels (console, HUD, dialogs) are game state — screen-space
    CharGridLayers in ``scene.grids`` — rendered generically by GridRenderer;
    this class knows nothing about their content. Dialogs open entirely
    game-side (see the dialogs package), so the window has no dialog code.
    """
    def __init__(self, dispatcher):
        self.canvas = vispy.scene.SceneCanvas()
        self.canvas.show()
        self.canvas.size = 1400, 900

        self.debug_line_of_sight = False
        self.debug_los_tex = False

        # feed canvas keyboard events into the game-side dispatcher
        self.input_dispatcher = dispatcher
        self.input_source = CanvasInputSource(self.canvas, dispatcher)

        # setup UI
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = Camera()
        self.view.camera.rect = [0, -5, 120, 60]
        self.view.camera.aspect = 0.6
        self.view.events.key_press.disconnect()

        self.camera_target = self.view.camera.rect
        self._last_camera_update = ptime.time()

        # when set, the next camera update jumps straight to the target rather
        # than easing toward it; see _level_changed
        self._snap_camera = False

        # game-state changes set this flag (from any thread); the frame tick
        # below turns it into one canvas repaint
        self._dirty = False
        self.grid_renderer = None

        self.frame_timer = vispy.app.Timer(start=True, connect=self._frame_tick, interval=0.016)

        self._follow_entity = None
        self.game_scene = None

    def attach_scene(self, scene):
        """Wire a game Scene to this window.

        Starts the generic grid renderer over ``scene.grids`` (which draws the
        console, HUD, and any open dialog grids alike). Dialogs, the command
        interpreter, and input handlers are all game-side and constructed by
        the caller (see return_to_carriage.py); the window only renders.
        """
        self.game_scene = scene
        self.grid_renderer = GridRenderer(self.canvas, scene, self.mark_dirty)
        scene.level_changed.connect(self._level_changed)

    def _level_changed(self):
        """Snap the camera on a level change instead of scrolling to it.

        The smooth scroll in _scroll_camera exists to follow a walking player.
        A level change teleports them, and easing across that gap would swoop
        the camera over a whole level of blackness before arriving.
        """
        self._snap_camera = True

    def mark_dirty(self):
        """Note that game state changed; a repaint follows on the frame tick.

        Just sets a flag, so it is safe to call from any thread — game-layer
        observers point here.
        """
        self._dirty = True

    def follow_entity(self, entity):
        if self._follow_entity is not None:
            self._follow_entity.location.global_changed.disconnect(self._update_camera_target)
        self._follow_entity = entity
        entity.location.global_changed.connect(self._update_camera_target)
        self._update_camera_target()

    def _frame_tick(self, ev):
        """60 Hz housekeeping on the GUI thread: camera scroll, grid sync,
        one coalesced repaint whenever game state changed, and the
        game-requested quit."""
        self._scroll_camera(ev)
        if self.grid_renderer is not None:
            self.grid_renderer.sync()
        if self._dirty:
            self._dirty = False
            self.canvas.update()
        if self.game_scene is not None and self.game_scene.quit_requested:
            self.canvas.close()

    def _scroll_camera(self, ev):
        now = ptime.time()
        dt = now - self._last_camera_update
        self._last_camera_update = now

        cr = vispy.geometry.Rect(self.view.camera.rect)
        tr = self.camera_target

        crv = np.array(cr.pos + cr.size, dtype='float32')
        trv = np.array(tr.pos + tr.size, dtype='float32')

        if not np.any(abs(trv - crv) > 1e-2):
            self._snap_camera = False
            return

        if self._snap_camera:
            self._snap_camera = False
            nrv = trv
        else:
            s = np.exp(-dt / 0.4)  # 400 ms settling time constant
            nrv = crv * s + trv * (1.0 - s)

        cr.pos = nrv[:2]
        cr.size = nrv[2:]
        self.view.camera.rect = cr

    def _update_camera_target(self, event=None):
        location = self._follow_entity.location
        pp = np.array(location.global_location.slot)
        cr = vispy.geometry.Rect(self.view.camera.rect)
        cc = np.array(cr.center)
        cs = np.array(cr.size)
        cp = np.array(cr.pos)

        dif = pp - cc
        maxdif = 0.1 * cs  # start correcting camera at 10% width from center
        for ax in (0, 1):
            if dif[ax] < -maxdif[ax]:
                cp[ax] += dif[ax] + maxdif[ax]
            elif dif[ax] > maxdif[ax]:
                cp[ax] += dif[ax] - maxdif[ax]

        cr.pos = cp
        self.camera_target = cr

    def quit(self):
        self.canvas.close()


class Camera(vispy.scene.cameras.PanZoomCamera):
    """Pan/zoom camera with all default keyboard interaction disabled"""
    def viewbox_key_event(self, ev):
        pass


