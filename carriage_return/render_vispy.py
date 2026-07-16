"""Vispy/OpenGL rendering backend for the game-owned render layers.

The game writes sprite data into layers (see layers.py); this module draws
them. Layer changes are detected by comparing integer version counters once
per draw, so an unchanged layer costs two comparisons per frame and a changed
layer costs whole-array numpy copies plus ranged VBO uploads.

Synchronization runs inside the visual's _prepare_draw so it covers both
on-screen draws and offscreen SceneCanvas.render() calls (which do not emit
canvas.events.draw).
"""
import vispy.scene

from .graphics import CharAtlas, SpritesVisual


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

    def sync(self):
        """Copy changed layer data into the visual; no-op when nothing changed."""
        glyphs = self.glyphs
        if glyphs.version != self._glyphs_version:
            new_chars = glyphs.chars[self._n_chars_synced:]
            if new_chars:
                # added in registry order, so glyph id == atlas index and the
                # layers' glyph arrays can be uploaded as sprite indices as-is
                self.atlas.add_chars(new_chars)
            self._n_chars_synced = len(glyphs.chars)
            self._glyphs_version = glyphs.version

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
