"""Generic vispy renderer for the scene's screen-space CharGridLayers.

One visual per ``scene.grids`` entry, created/destroyed by diffing the
LayerList's ``structure_version`` and re-uploaded when a grid's ``version``
changes — all from ``sync()``, which the window's frame tick calls on the
GUI thread. This module is meaning-free: borders, cursors and titles are
just cell data written by game-side painters.

All grid visuals share one CharAtlas fed incrementally from ``scene.glyphs``
in registry order, so glyph ids double as atlas indices (the same identity-
mapping contract render_vispy relies on; Sprites visuals listen for
``atlas_changed`` and re-upload the texture themselves).
"""
import numpy as np
import vispy.scene
import vispy.geometry

from .graphics import CharAtlas, Sprites


class _GridCamera(vispy.scene.cameras.PanZoomCamera):
    """PanZoomCamera with all user interaction disabled: grids don't scroll."""

    def viewbox_key_event(self, event):
        pass

    def viewbox_mouse_event(self, event):
        pass


class _GridVisual(object):
    """Vispy view of one CharGridLayer: a ViewBox holding a cell grid."""

    def __init__(self, canvas, grid, atlas, char_size):
        self.canvas = canvas
        self.grid = grid
        self.char_size = char_size

        self.view = vispy.scene.widgets.ViewBox(parent=canvas.scene)
        self.view.interactive = False
        self.view.padding = 0
        self.view.margin = 0
        self.view.camera = _GridCamera()

        self.txt = Sprites(atlas, sprite_size=(1, 1), point_cs='visual',
                           parent=self.view.scene)
        self.txt.update_gl_state(depth_test=False)
        self.sprites = self.txt.add_sprites(grid.shape)

        self._synced_version = None
        self._synced_structure = None
        self._configure()

    def _configure(self):
        """(Re)fit the visual to the grid's current shape: sprite region,
        cell positions, camera rect, and canvas placement."""
        rows, cols = self.grid.shape
        if self.sprites.shape != (rows, cols):
            self.sprites.set_shape((rows, cols))

        pos = np.zeros((rows, cols, 3), dtype='float32')
        # cell [r, c] at (x=c, y=rows-1-r): row 0 renders at the top
        pos[..., 0] = np.arange(cols)[np.newaxis, :]
        pos[..., 1] = (rows - 1 - np.arange(rows))[:, np.newaxis]
        self.sprites.position = pos

        # cell quads are centered on integer positions, so a half-cell margin
        # makes the grid fill the view exactly: one cell per char_size pixels
        self.view.camera.rect = vispy.geometry.Rect(-0.5, -0.5, cols, rows)
        self.place()

    def place(self):
        """Position the view per the grid's anchor/offset. The view always
        keeps one cell per char_size pixels: a grid larger than the canvas
        overflows the edge rather than compressing."""
        rows, cols = self.grid.shape
        cw, ch = self.char_size
        W, H = self.canvas.size
        w = cols * cw
        h = rows * ch
        off_rows, off_cols = self.grid.offset
        dx, dy = off_cols * cw, off_rows * ch

        parts = self.grid.anchor.split('-')
        if 'left' in parts:
            x = dx
        elif 'right' in parts:
            x = W - w - dx
        else:
            x = (W - w) / 2 + dx
        if 'top' in parts:
            y = dy
        elif 'bottom' in parts:
            y = H - h - dy
        else:
            y = (H - h) / 2 + dy
        self.view.rect = vispy.geometry.Rect(x, y, w, h)

    def sync(self):
        """Upload the grid's cell data if its version changed; refit the
        visual first if the grid was reshaped."""
        grid = self.grid
        if grid.structure_version != self._synced_structure:
            self._synced_structure = grid.structure_version
            self._configure()
            self._synced_version = None
        if grid.version == self._synced_version:
            return
        self._synced_version = grid.version
        self.sprites.sprite = grid.glyph
        self.sprites.fgcolor = grid.fgcolor
        self.sprites.bgcolor = grid.bgcolor
        self.txt.update()

    def close(self):
        self.view.parent = None


class GridRenderer(object):
    """Draws every screen-space CharGridLayer in ``scene.grids``.

    ``mark_dirty`` (a cheap, thread-safe callable — the window's dirty flag)
    subscribes to the LayerList's ``changed`` event and to each grid's, so
    any grid change schedules a frame; ``sync()`` then reconciles visuals
    with the list and uploads changed cell data on the GUI thread.
    """
    char_size = (10, 16)     # pixel size of one character cell (w, h)

    def __init__(self, canvas, scene, mark_dirty):
        self.canvas = canvas
        self.scene = scene
        self.mark_dirty = mark_dirty

        self.atlas = CharAtlas(size=12)
        self._n_chars_synced = 0

        self._visuals = {}   # CharGridLayer -> _GridVisual
        self._structure_version = None

        scene.grids.changed.connect(mark_dirty)
        canvas.events.resize.connect(self._canvas_resized)
        self._update_screen()
        self.sync()

    def sync(self):
        """Reconcile visuals with scene.grids and upload changed grids.

        GUI thread only (called from the frame tick); cheap when nothing
        changed.
        """
        # keep the shared atlas identical to the glyph registry. Advance by
        # exactly what was consumed — game threads may append to the registry
        # between the slice and the update, and len(registry.chars) would
        # silently skip those chars from the atlas forever.
        registry = self.scene.glyphs
        new_chars = registry.chars[self._n_chars_synced:]
        if new_chars:
            self.atlas.add_chars(new_chars)
            self._n_chars_synced += len(new_chars)

        grids = self.scene.grids
        if grids.structure_version != self._structure_version:
            self._structure_version = grids.structure_version
            wanted = [g for g in grids if g.space == 'screen']
            for grid in list(self._visuals):
                if grid not in wanted:
                    self._visuals.pop(grid).close()
                    grid.changed.disconnect(self.mark_dirty)
            for grid in wanted:
                if grid not in self._visuals:
                    grid.changed.connect(self.mark_dirty)
                    self._visuals[grid] = _GridVisual(self.canvas, grid,
                                                      self.atlas, self.char_size)

        for visual in self._visuals.values():
            visual.sync()

    def _update_screen(self):
        """Publish the canvas size in cells to the game (``scene.screen``).

        Runs on the GUI thread; the screen's subscriber (the Hud) reshapes and
        repaints its grids synchronously — pure numpy, so this is input
        flowing into game state, like a key event.
        """
        W, H = self.canvas.size
        cw, ch = self.char_size
        self.scene.screen.set_shape((int(H // ch), int(W // cw)))

    def _canvas_resized(self, event=None):
        self._update_screen()
        # reposition everything now; grids reshaped by the screen subscriber
        # are also refitted (and re-placed) at the next sync
        for visual in self._visuals.values():
            visual.place()
