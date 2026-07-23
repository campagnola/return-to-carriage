"""Spells cast into the world as short-lived, self-animating mobs.

A spell is an :class:`~.entity.Entity` that lives on one maze, draws its own
sprites straight into the scene's 'actors' layer, carries one or more point
lights, and advances in real time on its own daemon thread -- like a torch's
flame (see :class:`~.item.Torch`), it is game state that ticks on a background
clock and pushes each change out through ``scene.request_redraw()``. A spell is
*not* part of the turn system: it does not wait for the player to move, and it
never enters the maze inventory, so it neither blocks movement nor is "walked
on" -- it is pure spectacle that lights the room while it lasts.

Because a spell never joins the inventory, its light cannot follow a host
location the usual way; instead each light is *pinned* to a maze cell (see
:meth:`.light.Light.pin`). A moving spell re-pins its light every step. Pinning
drops the light's own caches but does not tell the level its lighting is stale,
so after moving (or after building/tearing down the lights) the spell nudges
the level to recomposite and repaint -- the same signal the flicker thread uses.

One module per spell, plus :mod:`.base` for the shared :class:`Spell` entity/
lights/anim-thread plumbing. Two projectile spells ship today, both cast in a
chosen cardinal direction:

- :class:`Fireball` -- a single very bright, slightly-white point light (about
  ten times a torch) that flies ``*`` across the map until it strikes a wall.
- :class:`Lightning` -- a string of ``/-\\|`` symbols that snaps into being
  along an almost-straight path, each symbol carrying a fierce cold-white point
  light, and vanishes again in under a second.

A third, ``glo`` (see :data:`SPELLS`), is not a mob at all: it is self-cast,
needs no direction, and toggles a persistent white light the player carries
(see :meth:`.player.Player.toggle_glo`).

Game-side package: no rendering library may be imported here.
"""
from .base import Spell
from .fireball import Fireball
from .glo import glo
from .lightning import Lightning


#: Arrow-key name -> unit ``(dx, dy)`` on the maze, matching the movement
#: handler's convention: x grows rightward, y grows upward (the maze image is
#: loaded flipped, so "up" is +y). The cast prompt collects one arrow key and
#: the interpreter maps it through here.
DIRECTIONS = {
    'Right': (1, 0),
    'Left': (-1, 0),
    'Up': (0, 1),
    'Down': (0, -1),
}


#: Spell name -> factory ``(scene, maze, pos, direction)``. Most build a mob
#: that flies in the given direction; a factory whose ``NEEDS_DIRECTION`` is
#: False (``glo``) ignores ``pos``/``direction`` and the prompt casts it the
#: moment its name resolves. The cast prompt matches typed text against these
#: names; "bal" -> fireball, "lit" -> lightning, "glo" -> glo (see dialogs.cast).
SPELLS = {
    'fireball': Fireball,
    'lightning': Lightning,
    'glo': glo,
}
