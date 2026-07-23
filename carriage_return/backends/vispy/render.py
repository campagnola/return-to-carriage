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

import numpy as np
import vispy.scene, vispy.gloo

from .graphics import CharAtlas, SpritesVisual, TextureMaskFilter, ShadowRenderer


class LayerSpritesVisual(SpritesVisual):
    """SpritesVisual that pulls pending game-layer data just before drawing.

    Draws the world's glyphs, which are tone-mapped against the sight field by
    a TextureMaskFilter, so it emits: its fragment shader stores each glyph's
    emission for that filter to add on top of the reflected light.
    """
    _layer_sync = None
    _field_sync = None
    emits = True

    def _prepare_draw(self, view):
        # Latch the glyph positions, then recompute the light field for that
        # same frame position immediately after. Reading both back to back --
        # with the heavy lighting compute happening AFTER both reads, not
        # between them -- is what keeps a glyph and its attached light on the
        # same cell: an input-thread move landing during the compute is picked
        # up whole by the next frame, never splitting one. See
        # VispySceneRenderer._sync_fields.
        if self._layer_sync is not None:
            self._layer_sync()
        if self._field_sync is not None:
            self._field_sync()
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
        # the subscriber only sets the window's dirty flag; the frame tick
        # turns a burst of writes into one repaint on the GUI thread. The
        # scene's sight FieldLayer must NOT be observed this way: it is
        # recomputed during every draw, so observing it would schedule draws
        # from within draws, forever.
        self.glyphs.changed.connect(ui.mark_dirty)
        for layer in self.layers:
            layer.changed.connect(ui.mark_dirty)

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
            region.emission = layer.emission
            self._synced_versions[layer.name] = versions


class VispySceneRenderer(object):
    """Complete vispy/OpenGL renderer for a Scene.

    Composes a VispyLayerRenderer for the sprite layers and adds the GL side
    of the visual-field pipeline:

    - constructs the GPU ShadowRenderer and injects it as the level's
      ``visibility``
    - drives ``level.update_sight(dt, player)`` once per canvas draw, on the
      level it has captured (never "whatever level is current")
    - uploads the level's ``light`` and ``memory_overlay`` FieldLayers to two
      textures (each only when its version changed) and applies them to the
      sprites as a mask filter that gates reflection/emission by line of sight

    The sight update runs from the sprite visual's _prepare_draw, right after
    the glyph positions are latched, so a frame draws the glyph and the light
    attached to it from one position read -- there is no one-frame lag between
    them even while an input thread moves the player concurrently. This is the
    only place the fields advance; it covers on-screen draws and offscreen
    SceneCanvas.render() alike.

    Time-based effects (eye adaptation, memory decay) advance by the real
    interval between frames, read through ``time_source``. That defaults to the
    monotonic wall clock and is overridden only by offscreen tooling that wants
    reproducible frames (see the ``time_source`` argument); the running game
    always uses the default.
    """
    def __init__(self, ui, scene, *, time_source=time.perf_counter):
        self.ui = ui
        self.scene = scene

        self.layer_renderer = VispyLayerRenderer(ui, scene.glyphs, list(scene.sprite_layers.values()))
        self.txt = self.layer_renderer.txt

        # the Level this renderer is currently set up for. Captured once by
        # _rebuild_for_level and read by every draw; the renderer never asks
        # the scene "what is the current level?" again, so a level switch on
        # another thread cannot change which level a frame in flight is drawing.
        self._level = None

        self.light_texture = None
        self.memory_texture = None
        self.sight_filter = None
        self._light_version = None
        self._memory_version = None
        self._last_update_time = None
        # Real time is read only through this callable. In the running game it
        # is the monotonic wall clock; offscreen tooling injects a deterministic
        # source so a rendered frame does not depend on machine speed. Nothing
        # in normal operation passes a non-default source.
        self._time_source = time_source

        # both the shadow renderer and the sight texture are sized from the
        # maze, so they are rebuilt on every level change (including the
        # initial build below, which scene.set_level already fired -- but the
        # renderer did not exist yet to hear it).
        scene.level_changed.connect(self._rebuild_for_level)
        self._rebuild_for_level()

        scene.redraw_requested.connect(ui.mark_dirty)

        # drive the per-frame field update from the sprite visual's pre-draw,
        # immediately after the glyphs are latched (see _prepare_draw), so the
        # glyph and its light are always read for the same frame position.
        self.txt._field_sync = self._sync_fields

    def _rebuild_for_level(self):
        """Re-create the maze-sized GL resources for the scene's current level.

        Called for the starting level and again whenever scene.set_level swaps
        in a differently-shaped maze. The current level is read exactly once,
        here, and everything below is sized from that captured level -- not
        from the scene, whose current level can change under a concurrent draw.
        """
        level = self._level = self.scene.level

        # GPU shadow-map provider for LOS/lighting computations, injected onto
        # the level so the player and its lights reach the provider sized to
        # their own maze -- never a provider left over from another level.
        level.visibility = ShadowRenderer(level.maze, self.ui.canvas,
                                          supersample=level.supersample)

        # light and memory fields -> textures, masking the sprites visual
        if self.sight_filter is not None:
            self.txt.detach(self.sight_filter)

        ms = level.maze.shape
        # Light: RGBA float, rgb carry raw linear HDR illuminance (may exceed 1,
        # ungated by line of sight), a carries the line-of-sight scalar (0..1).
        self.light_texture = vispy.gloo.Texture2D(shape=(*level.field_shape[:2], 4), format='rgba',
                                                  internalformat='rgba32f',
                                                  interpolation='linear', wrapping='repeat')
        # Memory: single-channel float, the display-space memory overlay. The
        # shader reads it as .r. r32f keeps the display-space value at full
        # precision; the filter samples .r regardless of channel count.
        self.memory_texture = vispy.gloo.Texture2D(shape=(*level.field_shape[:2], 1), format='red',
                                                   internalformat='r32f',
                                                   interpolation='linear', wrapping='repeat')
        # The tone map that turns these into displayable color, and the gating
        # of reflection/emission by line of sight, live in TextureMaskFilter
        # (per fragment).
        tr = self.txt.transforms.get_transform('framebuffer', 'visual')
        self.sight_filter = TextureMaskFilter(self.light_texture, self.memory_texture, tr,
                                              scale=(1./ms[1], 1./ms[0]))
        self.txt.attach(self.sight_filter)

        # force the next field sync to upload into the new textures
        self._light_version = None
        self._memory_version = None

    def _sync_fields(self):
        """Recompute and upload the visual fields for the current frame.

        Runs from the sprite visual's _prepare_draw, right after the glyph
        positions are latched, so the glyph and the light attached to it are
        read for one frame position. dt is the real interval since the previous
        frame, taken from ``self._time_source``.

        Composites the level this renderer was built for -- reached through the
        captured reference, so its light/memory fields, lighting and shadow
        provider are all the one maze's shape. When the player is not on that
        level (the window between a level switch and the player being moved onto
        it), Level.update_sight blocks the view and the level's remembered field
        is what shows.
        """
        now = self._time_source()
        dt = 0.0 if self._last_update_time is None else now - self._last_update_time
        self._last_update_time = now

        level = self._level
        level.update_sight(dt, self.scene.player)

        # exposure tracks the player's eye adaptation and changes every frame,
        # so push it to the tone-mapping filter each draw (cheap uniform set).
        # The field textures themselves only re-upload on a version change
        # below. When there is no player, keep the last exposure -- line of
        # sight is 0 in that case anyway, so reflection and emission are gated
        # off on the GPU and only the memory overlay shows.
        player = self.scene.player
        if player is not None:
            self.sight_filter.set_exposure(player.adaptation.exposure)
            # Adaptation is time-based, so it must keep advancing even when the
            # player stands still and nothing else marks the canvas dirty. While
            # the eye is still settling, ask for another frame; this self-limits
            # -- once adaptation reaches its target (settling clears) the scene
            # goes idle again. mark_dirty only sets a flag the 60 Hz frame tick
            # consumes, so this is not the forbidden "schedule a draw inside a
            # draw" (which is why the sight field itself is never observed).
            if player.adaptation.settling:
                self.ui.mark_dirty()

        light = level.light
        if light.version != self._light_version:
            self.light_texture.set_data(light.data)
            self._light_version = light.version

        memory = level.memory_overlay
        if memory.version != self._memory_version:
            self.memory_texture.set_data(memory.data[..., np.newaxis])
            self._memory_version = memory.version
