import numpy as np
import vispy.scene, vispy.app
import vispy.util.ptime as ptime

from .graphics import CharAtlas, Sprites
from .input import InputDispatcher, CommandInputHandler
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
        self.view.camera = Camera()
        self.view.camera.rect = [0, -5, 120, 60]
        self.view.camera.aspect = 0.6
        self.view.events.key_press.disconnect()

        self.camera_target = self.view.camera.rect
        self._last_camera_update = ptime.time()
        self.scroll_timer = vispy.app.Timer(start=True, connect=self._scroll_camera, interval=0.016)

        self.console_grid = self.canvas.central_widget.add_grid()

        self.stats_box = TextBox((2, 160))
        self.console_grid.add_widget(self.stats_box.view, 1, 0, 1, 2)
        self.stats_box.write(
            "HP:17/33   Food:56%  Water:34%  Sleep:65%   Weight:207(45)    Level:3  Int:12  Str:9  Wis:11  Cha:2")
        self.stats_box.view.height_max = 30
        self.stats_box.view.stretch = (1, 10)

        self.info_box = TextBox((15, 80))
        self.console_grid.add_widget(self.info_box.view, 2, 0)
        self.info_box.write("There is a scroll of infinite recursion here.")
        self.info_box.view.height_max = 200
        self.stats_box.view.stretch = (1, 1)

        self.console = Console((15, 80))
        self.console_grid.add_widget(self.console.view, 2, 1)

        self.console.view.height_max = 200

        self.console.write('Hello?')
        self.console.write('Is anybody\n    there?')
        # self.console.view.camera.rect = [-1, -1, 30, 3]

        self.command = CommandInterpreter(self)
        self.cmd_input_handler = CommandInputHandler(self.console, self.command)

        self._follow_entity = None

    def follow_entity(self, entity):
        if self._follow_entity is not None:
            self._follow_entity.location.global_changed.disconnect(self._update_camera_target)
        self._follow_entity = entity
        entity.location.global_changed.connect(self._update_camera_target)
        self._update_camera_target()

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


class TextBox(object):
    def __init__(self, shape):
        self.shape = shape

        self.view = vispy.scene.widgets.ViewBox(border_color=(1, 1, 1, 0.2), bgcolor=(0, 0, 0, 0.4))
        self.view.camera = Camera()
        self.view.camera.rect = vispy.geometry.Rect(-0.7, -0.7, shape[1], shape[0])
        self.view.padding = 0
        self.view.margin = 1

        # generate a texture for each character we need
        self.atlas = CharAtlas(size=12)
        ascii_chars = [chr(i) for i in range(0x20, 128)]
        self.atlas.add_chars(ascii_chars)

        # create sprites visual
        self.txt = Sprites(self.atlas, sprite_size=(1, 1), point_cs='visual', parent=self.view.scene)
        self.txt.update_gl_state(depth_test=False)

        self.txt_sprites = self.txt.add_sprites(shape)
        self.txt_sprites.sprite = 0

        pos = np.zeros(shape + (3,), dtype='float32')
        pos[...,:2] = np.mgrid[0:shape[1], 0:shape[0]].transpose(2, 1, 0)
        self.txt_sprites.position = pos

        fgcolor = np.ones(shape + (4,), dtype='float32')
        fgcolor[...,3] = 0.5
        self.txt_sprites.fgcolor = fgcolor

        bgcolor = np.ones(shape + (4,), dtype='float32')
        bgcolor[..., 3] = 0
        self.txt_sprites.bgcolor = bgcolor

        self.lines = []

    def write(self, txt):
        self.lines.extend(txt.split('\n'))
        self.update_text()

    def set_last_line(self, line):
        self.lines[-1] = line
        self.update_text()

    def remove_last_line(self):
        self.lines.pop(-1)
        self.update_text()

    def update_text(self):
        sprites = np.zeros(self.shape, dtype='uint8')
        sprites[:] = ord(' ')
        for i in range(min(self.shape[0], len(self.lines))):
            # todo: line wrapping
            line = np.fromstring(self.lines[-i-1].encode('ascii'), dtype='ubyte')
            sprites[i, :len(line)] = line[:self.shape[1]]
        self.txt_sprites.sprite = sprites - 0x20


class Console(TextBox):
    def __init__(self, shape):
        TextBox.__init__(self, shape)
        self.view.stretch = (1, 10)
        # self.console.view.parent = self.canvas.scene
        self.view.rect = vispy.geometry.Rect(30, 620, 1350, 250)
        self.transform = vispy.visuals.transforms.STTransform((0, 0, -0.5))
        # self.console.view.camera.aspect = 0.6




class Camera(vispy.scene.cameras.PanZoomCamera):
    """Pan/zoom camera with all default keyboard interaction disabled"""
    def viewbox_key_event(self, ev):
        pass


