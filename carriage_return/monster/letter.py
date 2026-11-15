"""Letter monsters: a monster drawn as a single letter of the alphabet."""
import string

from .base import Monster

#: Glyphs a letter monster may take: A-Z and a-z, less the letters other
#: entities already use, so a monster is never mistaken for one -- 't' is a
#: torch (:class:`~.item.Torch`), 'O' a hole (:class:`~.portal.Hole`).
LETTERS = ''.join(c for c in string.ascii_letters if c not in 'tO')


class Letter(Monster):
    """A monster that is one letter of the alphabet, *letter*."""

    name = "letter"

    def __init__(self, letter, location, scene, behaviour=None):
        self.char = letter
        Monster.__init__(self, location, scene, behaviour=behaviour, obj_name=letter)

    @property
    def description(self):
        return "letter %s" % self.char
