# coding: utf8


class Monster(object):
    """A mob standing on one maze.

    *maze* defaults to whatever level the scene is showing when the monster is
    built. It matters because the sprite layers are shared by the whole world:
    a monster on a level that is not displayed must hide, or it would go on
    being drawn at its own level's coordinates over whatever the player is
    actually looking at.
    """
    def __init__(self, position, scene, maze=None):
        self._position = None
        self.scene = scene
        self.maze = maze if maze is not None else scene.maze

        self.sprite = scene.sprite_layers['actors'].add_sprites((1,))
        self.sprite.fgcolor = (0.6, 0.6, 0.6, 1)
        self.sprite.bgcolor = (0, 0, 0, 1)
        self.sprite.glyph = scene.glyphs[u'Y']

        self.position = position
        scene.level_changed.connect(self._level_changed)

    @property
    def on_current_level(self):
        return self.maze is self.scene.maze

    def take_turn(self):
        if not self.on_current_level:
            return
        l = list(self.position)
        l[1] -= 1
        self.position = l

    def _level_changed(self):
        self._draw()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        old_pos = self._position
        self._position = pos
        self._draw()
        self.scene.monster_moved(self, old_pos)

    def _draw(self):
        if self.on_current_level:
            self.sprite.position = (self._position[0], self._position[1], -0.1)
        else:
            self.sprite.position = (float('nan'),) * 3
