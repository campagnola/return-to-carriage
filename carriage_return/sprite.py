

class SingleCharSprite:
    """Component that draws a single-character sprite.

    - Reacts to location changes
    - Hides when not visible
    - Assumes background color from maze
    """
    def __init__(self, entity, zval, char, fg_color=(1, 1, 1, 1), bg_color=None):
        self.zval = zval
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.entity = entity
        self.sprite = entity.scene.txt.add_sprites((1,))
        self.sprite.sprite = entity.scene.atlas.add_chars(char)
        self.sprite.fgcolor = fg_color
        self.sprite.bgcolor = bg_color

        entity.location.global_changed.connect(self._update_location)

    def _update_location(self, event):
        container = self.entity.location.container
        if container.type.isa('maze'):
            pos = self.entity.location.slot
            self.sprite.position = pos + (self.zval,)
            self.sprite.bgcolor = self.bg_color or container.bg_color[pos[1], pos[0]]
        else:
            self.sprite.position = (float('nan'),) * 3
