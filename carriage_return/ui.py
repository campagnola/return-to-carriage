import numpy as np
import vispy.scene, vispy.app
import vispy.util.ptime as ptime

from .input import InputDispatcher, CommandInputHandler
from .graphics import Console
from .console import CommandInterpreter


class MainWindow:
    """Implements user interface: graphical panels, key input handling
    """
    def __init__(self):
        self.canvas = vispy.scene.SceneCanvas()
        self.canvas.show()
        self.canvas.size = 1400, 900

        self.debug_line_of_sight = False
        self.debug_los_tex = False

        # Setup input event handling
        self.input_dispatcher = InputDispatcher(self.canvas)
        self.command_mode = False

        # setup UI
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.rect = [0, -5, 120, 60]
        self.view.camera.aspect = 0.6

        self.camera_target = self.view.camera.rect
        self._last_camera_update = ptime.time()
        self.scroll_timer = vispy.app.Timer(start=True, connect=self._scroll_camera, interval=0.016)

        self.console_grid = self.canvas.central_widget.add_grid()

        self.stats_box = Console((2, 160))
        self.console_grid.add_widget(self.stats_box.view, 1, 0, 1, 2)
        self.stats_box.write(
            "HP:17/33   Food:56%  Water:34%  Sleep:65%   Weight:207(45)    Level:3  Int:12  Str:9  Wis:11  Cha:2")
        self.stats_box.view.height_max = 30
        self.stats_box.view.stretch = (1, 10)

        self.info_box = Console((15, 80))
        self.console_grid.add_widget(self.info_box.view, 2, 0)
        self.info_box.write("There is a scroll of infinite recursion here.")
        self.info_box.view.height_max = 200
        self.stats_box.view.stretch = (1, 1)

        self.console = Console((15, 80))
        self.console_grid.add_widget(self.console.view, 2, 1)
        self.console.view.stretch = (1, 10)
        # self.console.view.parent = self.canvas.scene
        self.console.view.rect = vispy.geometry.Rect(30, 620, 1350, 250)
        self.console.transform = vispy.visuals.transforms.STTransform((0, 0, -0.5))
        # self.console.view.camera.aspect = 0.6

        self.console.view.height_max = 200

        self.console.write('Hello?')
        self.console.write('Is anybody\n    there?')
        self.console.write(''.join([chr(i) for i in range(0x20, 128)]))
        # self.console.view.camera.rect = [-1, -1, 30, 3]

        self.command = CommandInterpreter(self)
        self.cmd_input_handler = CommandInputHandler(self.console, self.command)

    def toggle_command_mode(self):
        # todo: visual cue
        self.command_mode = not self.command_mode
        if self.command_mode:
            self.cmd_input_handler.activate()
        else:
            self.cmd_input_handler.deactivate()

    def _scroll_camera(self, ev):
        now = ptime.time()
        dt = now - self._last_camera_update
        self._last_camera_update = now

        cr = vispy.geometry.Rect(self.view.camera.rect)
        tr = self.camera_target

        crv = np.array(cr.pos + cr.size, dtype='float32')
        trv = np.array(tr.pos + tr.size, dtype='float32')

        if not np.any(abs(trv - crv) > 1e-2):
            return

        s = np.exp(-dt / 0.4)  # 400 ms settling time constant
        nrv = crv * s + trv * (1.0 - s)

        cr.pos = nrv[:2]
        cr.size = nrv[2:]
        self.view.camera.rect = cr

    def _update_camera_target(self, event):
        location = event.source
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
