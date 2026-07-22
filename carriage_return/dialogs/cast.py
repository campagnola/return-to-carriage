"""The spell-casting dialog: model, key loop, and painter in one place.

Casting is a two-phase modal prompt. First the player types a spell name and
presses Enter; the typed text is matched against the known spell names as a
substring, so "bal" finds *fireball* and "lit" finds *lightning*. Then the
prompt waits for a single arrow key -- the direction to cast in -- and returns
``(spell_name, arrow_key)``. Escape cancels at either phase (result None).

``CastPrompt`` is the pure-data model, ``run_cast`` the sequential key loop a
DialogSession runs on its own thread, and ``CastPainter`` renders it into a
screen-space grid. The interpreter turns the returned arrow key into a maze
direction (see ``spell.DIRECTIONS``) and builds the spell. See base.py for the
change-tracking and painter contracts.
"""
from ..input import KeyPress
from .base import CharGridPainter, Widget


#: Arrow keys accepted in the direction phase. The dialog only validates that a
#: direction was chosen; mapping the key to a maze vector is the caller's job
#: (spell.DIRECTIONS), so the dialog need not import the spell module.
ARROWS = ('Up', 'Down', 'Left', 'Right')


def _subsequence(typed, name):
    """True if *typed*'s characters appear in *name* in order (not necessarily
    contiguously): "bal" -> "fireball", "lit" -> "lightning"."""
    it = iter(name)
    return all(ch in it for ch in typed)


class CastPrompt(Widget):
    """Two-phase casting model: a name to type, then a direction to choose.

    ``phase`` is 'name' while the player types (``text`` accumulates) and
    'direction' once a spell has been resolved (``spell_name`` set). ``message``
    carries a one-line note back to the player -- a fizzled mumble for an
    unknown name, or a disambiguation when the text matches more than one spell.
    """

    def __init__(self, spells):
        Widget.__init__(self)
        self.spells = spells        # {name: factory}
        self.text = ""
        self.phase = 'name'
        self.spell_name = None
        self.message = ""

    def type_char(self, ch):
        self.text += ch
        self._changed()

    def backspace(self):
        self.text = self.text[:-1]
        self._changed()

    def resolve(self):
        """Match the typed text against the spell names; advance on a hit.

        Matching is by in-order subsequence, not contiguous substring, so a few
        characteristic letters name a spell: "bal" finds *fireball* and "lit"
        finds *lightning*. Returns the matched name (and moves to the direction
        phase) when exactly one spell matches; otherwise sets ``message``, stays
        in the name phase, and returns None.
        """
        typed = self.text.strip().lower()
        matches = [name for name in self.spells
                   if typed and _subsequence(typed, name)]
        if len(matches) == 1:
            self.spell_name = matches[0]
            self.phase = 'direction'
            self.message = ""
            self._changed()
            return matches[0]
        if not matches:
            self.message = ('You mumble "%s" -- nothing happens.'
                            % (self.text.strip() or '...'))
        else:
            self.message = "Did you mean %s?" % " or ".join(sorted(matches))
        self._changed()
        return None


def run_cast(session, prompt):
    """Key handling for a CastPrompt; returns ``(spell_name, arrow_key)`` or None.

    Name phase: printable characters build the name, Backspace edits, Enter
    resolves it. Direction phase: the next arrow key finishes. Escape cancels at
    either phase. Key releases and unknown keys are ignored.
    """
    while True:
        event = session.get()
        if not isinstance(event, KeyPress):
            continue
        if event.key == 'Escape':
            return None
        if prompt.phase == 'name':
            if event.key in ('Enter', 'Return'):
                prompt.resolve()
            elif event.key == 'Backspace':
                prompt.backspace()
            elif event.text and len(event.text) == 1 and event.text.isprintable():
                prompt.type_char(event.text)
        elif event.key in ARROWS:
            return (prompt.spell_name, event.key)


class CastPainter(CharGridPainter):
    """Paints a CastPrompt: title, the current phase's prompt line, a message,
    and key hints."""

    WIDTH = 46

    def _content_shape(self):
        # title, blank, prompt, message, hint
        return 5, self.WIDTH

    def _render_content(self):
        p = self.model
        self._write_line(0, "Cast a spell", fg=self.TITLE_FG)
        if p.phase == 'name':
            self._write_line(2, "Spell: " + p.text + "_")
            hint = "Type a name, Enter to cast, Esc to cancel"
        else:
            self._write_line(2, "%s -- which way?" % p.spell_name.capitalize())
            hint = "Press an arrow key  (Esc to cancel)"
        if p.message:
            self._write_line(3, p.message, fg=self.HINT_FG)
        self._write_line(self.nrows - 1, hint, fg=self.HINT_FG)
