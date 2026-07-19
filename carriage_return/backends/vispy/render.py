"""Vispy/OpenGL rendering backend for the game-owned render layers.

The game writes sprite data into layers (see layers.py); this module draws
them. Layer changes are detected by comparing integer version counters once
per draw, so an unchanged layer costs two comparisons per frame and a changed
layer costs whole-array numpy copies plus ranged VBO uploads.

Synchronization runs inside the visual's _prepare_draw so it covers both
on-screen draws and offscreen SceneCanvas.render() calls (which do not emit
canvas.events.draw).
"""
import time

import vispy.scene, vispy.gloo

from .graphics import CharAtlas, SpritesVisual, TextureMaskFilter, ShadowRenderer


class LayerSpritesVisual(SpritesVisual):
    """SpritesVisual that pulls pending game-layer data just before drawing."""
    _layer_sync = None

    def _prepare_draw(self, view):
        if self._layer_sync is not None:
            self._layer_sync()
        return SpritesVisual._prepare_draw(self, view)


LayerSprites = vispy.scene.visuals.create_visual_node(LayerSpritesVisual)


class VispyLayerRenderer(object):
    """Draws a GlyphRegistry + SpriteLayers using a single Sprites visual.

    Each layer maps to one SpriteData region in the visual; regions are
    created lazily (a layer that never gains sprites is never uploaded) and
    resized when the layer's sprite count changes. Depth ordering between
    layers comes from the z coordinate of sprite positions, exactly as it did
    when entities wrote into the visual directly.
    """
    def __init__(self, ui, glyphs, layers):
        self.glyphs = glyphs
        self.layers = list(layers)

        self.atlas = CharAtlas()
        self._glyphs_version = None
        self._n_chars_synced = 0

        self.txt = LayerSprites(self.atlas, sprite_size=(1, 1), point_cs='visual',
                                parent=ui.view.scene)
        self.txt._layer_sync = self.sync

        self._regions = {layer.name: None for layer in self.layers}
        self._synced_versions = {layer.name: None for layer in self.layers}

        # schedule a redraw whenever the game writes to a layer. Layer writes
        # can come from worker threads (gamepad input, dialog threads), so
        # the observer only sets the window's dirty flag; the frame tick
        # turns a burst of writes into one repaint on the GUI thread. The
        # scene's sight FieldLayer must NOT be observed this way: it is
        # recomputed during every draw, so observing it would schedule draws
        # from within draws, forever.
        self.glyphs.observer = ui.mark_dirty
        for layer in self.layers:
            layer.observer = ui.mark_dirty

    def sync(self):
        """Copy changed layer data into the visual; no-op when nothing changed."""
        glyphs = self.glyphs
        if glyphs.version != self._glyphs_version:
            self._glyphs_version = glyphs.version
            new_chars = glyphs.chars[self._n_chars_synced:]
            if new_chars:
                # added in registry order, so glyph id == atlas index and the
                # layers' glyph arrays can be uploaded as sprite indices as-is.
                # Advance by exactly what was consumed — game threads may
                # append to the registry between the slice and this line, and
                # len(glyphs.chars) would silently skip those chars forever.
                self.atlas.add_chars(new_chars)
                self._n_chars_synced += len(new_chars)

        for layer in self.layers:
            versions = (layer.version, layer.structure_version)
            if versions == self._synced_versions[layer.name]:
                continue

            region = self._regions[layer.name]
            if region is None:
                if len(layer) == 0:
                    continue
                region = self.txt.add_sprites((len(layer),))
                self._regions[layer.name] = region
            elif len(region) != len(layer):
                region.set_shape((len(layer),))

            region.position = layer.position
            region.sprite = layer.glyph
            region.fgcolor = layer.fgcolor
            region.bgcolor = layer.bgcolor
            self._synced_versions[layer.name] = versions


class VispySceneRenderer(object):
    """Complete vispy/OpenGL renderer for a Scene.

    Composes a VispyLayerRenderer for the sprite layers and adds the GL side
    of the visual-field pipeline:

    - constructs the GPU ShadowRenderer and injects it as scene.visibility
    - drives scene.update_sight(dt) once per canvas draw
    - uploads the scene's ``sight`` FieldLayer to a texture (only when its
      version changed) and applies it to the sprites as a mask filter

    As in the pre-split design, the sight update runs as a canvas draw-event
    callback, i.e. after the scene has been drawn; the updated field is
    rendered on the next frame. Offscreen SceneCanvas.render() calls do not
    emit draw events, so batch/screenshot code must call update() explicitly
    (with an explicit dt for determinism).
    """
    def __init__(self, ui, scene):
        self.ui = ui
        self.scene = scene

        self.layer_renderer = VispyLayerRenderer(ui, scene.glyphs, list(scene.sprite_layers.values()))
        self.txt = self.layer_renderer.txt

        # GPU shadow-map provider for LOS/lighting computations
        scene.visibility = ShadowRenderer(scene.maze, ui.canvas, supersample=scene.supersample)

        # sight field -> texture, masking the sprites visual
        ms = scene.maze.shape
        self.sight_texture = vispy.gloo.Texture2D(shape=scene.field_shape, format='rgb',
                                                  interpolation='linear', wrapping='repeat')
        tr = self.txt.transforms.get_transform('framebuffer', 'visual')
        self.sight_filter = TextureMaskFilter(self.sight_texture, tr, scale=(1./ms[1], 1./ms[0]))
        self.txt.attach(self.sight_filter)

        self._sight_version = None
        self._last_update_time = None

        scene.redraw_observer = ui.mark_dirty

        ui.canvas.events.draw.connect(self._on_draw)

    def _on_draw(self, event):
        now = time.perf_counter()
        dt = 0.0 if self._last_update_time is None else now - self._last_update_time
        self._last_update_time = now
        self.update(dt)

    def update(self, dt):
        """Advance the scene's visual-field state by *dt* seconds and upload
        the result if it changed."""
        self.scene.update_sight(dt)
        sight = self.scene.sight
        if sight.version != self._sight_version:
            self.sight_texture.set_data(sight.data)
            self._sight_version = sight.version
