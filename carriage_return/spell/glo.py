"""The ``glo`` spell: a self-cast light the player carries.

Game-side module: no rendering library may be imported here.
"""


def glo(scene, maze, pos, direction):
    """Cast 'glo': toggle the player's own bright white light on or off.

    Unlike the projectile spells, glo needs no direction (``pos``/``direction``
    are ignored) and pins nothing to the map: it toggles a persistent white
    light the player carries (see :meth:`.player.Player.toggle_glo`), so it
    needs no inventory slot and lasts until the spell is cast again.
    """
    scene.player.toggle_glo()


#: glo is self-cast on the player, so the prompt skips the direction phase.
glo.NEEDS_DIRECTION = False
