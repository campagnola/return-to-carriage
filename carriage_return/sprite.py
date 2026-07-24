from carriage_return.entity import Component


class SingleCharSprite(Component):
    """Component that draws a single-character sprite.

    - Reacts to location changes
    - Hides when not visible
    - Assumes background color from maze

    Writes into one of the scene's game-owned sprite layers; no rendering
    library is touched here.

    The sprite layers are shared by the whole world, not per level, so an
    entity standing on a level that is not currently displayed must hide
    itself -- otherwise the dungeon's torches would go on being drawn at their
    dungeon coordinates while the player walks the sewer. Hence the subscription
    to ``scene.level_changed``: every sprite re-checks which maze it is on
    whenever the visible level changes.
    """
    def __init__(self, entity, zval, char, fg_color=(1, 1, 1, 1), bg_color=None,
                 fg_emission=None, layer='items'):
        Component.__init__(self, entity, component_type='single_char_sprite')
        self.zval = zval
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.scene = entity.scene
        self.sprite = entity.scene.sprite_layers[layer].add_sprites((1,))
        self.sprite.glyph = entity.scene.glyphs[char]
        self.sprite.fgcolor = fg_color
        self.sprite.bgcolor = bg_color
        # Light the glyph emits on its own (RGB, linear, same units as the
        # illuminance the light field carries), added on top of what it
        # reflects. None/absent means a plain reflective glyph.
        self.sprite.fg_emission = (0, 0, 0) if fg_emission is None else fg_emission

        entity.location.global_changed.connect(self._update_location)
        entity.scene.level_changed.connect(self._level_changed)

    def _level_changed(self):
        self._update_location(None)

    def _update_location(self, event):
        container = self.parent_entity.location.container
        if container is not None and container.type.isa('maze') and container is self.scene.maze:
            pos = self.parent_entity.location.slot
            self.sprite.position = pos + (self.zval,)
            self.sprite.bgcolor = self.bg_color or container.bg_color[pos[1], pos[0]]
        else:
            self.hide()

    def hide(self):
        self.sprite.position = (float('nan'),) * 3
