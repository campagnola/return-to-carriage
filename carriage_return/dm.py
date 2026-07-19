

class DungeonMaster:
    """Responsible for managing turns, accepting requests to change the world state, and
    deciding what actual changes to make.
    """
    def __init__(self, scene):
        self.scene = scene

    def request_player_move(self, player, newpos):
        """Attempt to move the player to newpos.
        """
        newpos = newpos.astype(int)
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

    def move_player(self, player, pos):
        player.location.update(self.scene.maze, pos)
        self.end_turn()

        # Holes and doors act on arrival; stairs wait to be used (see
        # world.PortalEnd.command). Checked after end_turn so the step that
        # brought the player here is a complete turn in its own right.
        end = self.portal_end_at(player.location.slot)
        if end is not None and end.walk_on:
            self.traverse(player, end)

    def use_stairs(self, player, command):
        """Act on the ``<`` or ``>`` command at the player's feet."""
        end = self.portal_end_at(player.location.slot)
        if end is None or end.command != command:
            direction = "up" if command == '<' else "down"
            self.scene.write("There are no stairs %s here." % direction)
            return
        self.traverse(player, end)

    def portal_end_at(self, pos):
        """The PortalEnd at *pos* on the current level, or None."""
        if self.scene.world is None:
            return None
        return self.scene.world.portal_end_at(self.scene.maze, pos)

    def traverse(self, player, from_end):
        """Take *player* through the portal *from_end* belongs to.

        Refuses when the portal cannot be entered from this side -- which is
        what makes the sewer's ceiling opening a thing you stand under rather
        than a way back home.
        """
        if not from_end.enterable:
            self.scene.write(self.refusal(from_end))
            return False

        to_end = from_end.portal.other(from_end)
        self.scene.set_level(to_end.level)
        player.location.update(to_end.level.maze, to_end.pos)
        return True

    def refusal(self, end):
        """The message for a portal end that cannot be entered from this side."""
        if end.blocktype == 'hole':
            return "The opening is far above you; there is no way back up."
        return "You cannot go that way."

    def end_turn(self):
        for mlist in list(self.scene.monsters.values()):
            for m in mlist:
                m.take_turn()
