# coding: utf8
"""The :class:`Scroll` item: a readable scroll of infinite recursion."""
from .base import Item


class Scroll(Item):

    name = "scroll of infinite recursion"
    char = u'次'
    readable = True
    takeable = True
    mass = 0.05
    length = 20.0
    fg_color = (0.8, 0.8, 0.8, 1.0)

    # page contents shown by the 'read' action (opened as a pager below)
    pages = [
        "Instructions for use:\n"
        "\n"
        "1. Carefully unroll the scroll.\n"
        "2. Read the instructions for use.",

        "3. If the instructions are unclear,\n"
        "   consult the instructions.\n"
        "4. Repeat until enlightened.",

        "You put the scroll down, none the\n"
        "wiser but strangely satisfied.",
    ]

    def read(self, reader):
        """Show the scroll's pages in a modal pager (the joke ends on the last
        page; the scroll is not consumed). Returns the pager session."""
        from .. import dialogs
        return dialogs.open_pager(self.scene, self.description, self.pages)
