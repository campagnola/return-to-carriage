

class DungeonMaster:
    """Responsible for managing turns, accepting requests to change the world state, and
    deciding what actual changes to make.
    """
    def __init__(self, scene):
        self.scene = scene

    def request_player_move(self, player, newpos):
        """Attempt to move the player to newpos.
        """
        pos = player.location.slot
        j, i = newpos
        j0, i0 = player.location.slot
        if self.scene.maze.blocktype_at(i, j)['walkable']:
            self.move_player(player, newpos)
        elif self.scene.maze.blocktype_at(i0, j)['walkable']:
            newpos[1] = i0
            self.move_player(player, newpos)
        elif self.scene.maze.blocktype_at(i, j0)['walkable']:
            newpos[0] = j0
            self.move_player(player, newpos)
        self.norm_light = None

    def move_player(self, player, pos):
        player.location.update(self.scene.maze, pos)
        self.end_turn()

    def end_turn(self):
        for mlist in list(self.scene.monsters.values()):
            for m in mlist:
                m.take_turn()

