

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
        j, i = newpos
        j0, i0 = player.location.slot
        if self.walkable((j, i)):
            self.move_player(player, newpos)
        elif self.walkable((j, i0)):
            newpos[1] = i0
            self.move_player(player, newpos)
        elif self.walkable((j0, i)):
            newpos[0] = j0
            self.move_player(player, newpos)

    def walkable(self, pos):
        """True if a mover may stand on cell *pos* ``(x, y)`` of the current maze.

        The terrain must be walkable *and* nothing standing on the cell may
        block it: walkability is aggregated over the ground and every entity
        there, so a shut door stops you on otherwise-open floor without the
        terrain grid knowing anything about it.
        """
        x, y = pos
        maze = self.scene.maze
        if not maze.blocktype_at(y, x)['walkable']:
            return False
        return not any(e.blocks_movement for e in maze.inventory[(x, y)])

    def move_player(self, player, pos):
        player.location.update(self.scene.maze, pos)
        self.end_turn()

        # Ask whatever the player stepped onto what happens -- the dungeon
        # master does not know a hole from a scroll, it just asks each entity on
        # the cell. An entity may call back to request an action (a hole asks to
        # traverse); if that carries the player off this cell there is nothing
        # more here to react, so stop. Checked after end_turn so the step that
        # brought the player here is a complete turn in its own right.
        here = player.location.place
        maze, cell = here
        for entity in list(maze.inventory[cell]):
            if entity is player:
                continue
            entity.on_walked_on(player, self)
            if player.location.place != here:
                break

    def use_stairs(self, player, command):
        """Apply the ``<`` / ``>`` command at the player's feet.

        The dungeon master does not know what a stair is: it asks each entity on
        the cell whether it responds to the command and stops at the first that
        does. If nothing does, it says so.
        """
        for entity in list(self.scene.maze.inventory[tuple(player.location.slot)]):
            if entity is player:
                continue
            if entity.on_command(player, command, self):
                return
        direction = "up" if command == '<' else "down"
        self.scene.write("There are no stairs %s here." % direction)

    def request_traverse(self, mover, from_end):
        """A portal end's request to send *mover* through to its far side.

        A request, not a decision: the dungeon master has final say. A side that
        cannot be entered is refused, with the end's own message; otherwise the
        mover crosses to the far end, switching levels. Returns whether it
        happened. Called *by the entity* (see :meth:`.portal.PortalEnd`), not by
        any type-inspecting code here.
        """
        if not from_end.enterable:
            self.scene.write(from_end.refusal)
            return False

        to_end = from_end.portal.other(from_end)
        self.scene.set_level(to_end.level)
        mover.location.update(to_end.level.maze, to_end.pos)
        return True

    def end_turn(self):
        for mlist in list(self.scene.monsters.values()):
            for m in mlist:
                m.take_turn()
