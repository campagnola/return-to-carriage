"""The spell-casting dialog: interactive state, key loop, and drawing in one widget.

Casting is a two-phase modal prompt. First the player types a spell name and
presses Enter; the typed text is matched against the known spell names as a
substring, so "bal" finds *fireball* and "lit" finds *lightning*. Then the
prompt waits for a single arrow key -- the direction to cast in -- and returns
``(spell_name, arrow_key)``. Escape cancels at either phase (result None).

``CastWidget`` is the ``carriage_return.widgets.Widget`` holding this state
and its own rendering; ``run_cast`` is the sequential key loop a
DialogSession runs on its own thread. The interpreter turns the returned
arrow key into a maze direction (see ``spell.DIRECTIONS``) and builds the
spell.
"""
from ..input import KeyPress
from ..widgets import FG, BG, HINT_FG, TITLE_FG, Widget


#: Arrow keys accepted in the direction phase. The dialog only validates that a
#: direction was chosen; mapping the key to a maze vector is the caller's job
#: (spell.DIRECTIONS), so the dialog need not import the spell module.
ARROWS = ('Up', 'Down', 'Left', 'Right')


def _subsequence(typed, name):
    """True if *typed*'s characters appear in *name* in order (not necessarily
    contiguously): "bal" -> "fireball", "lit" -> "lightning"."""
    it = iter(name)
    return all(ch in it for ch in typed)


class CastWidget(Widget):
    """Two-phase casting prompt: a name to type, then a direction to choose.

    ``phase`` is 'name' while the player types (``text`` accumulates) and
    'direction' once a spell has been resolved (``spell_name`` set). ``message``
    carries a one-line note back to the player -- a fizzled mumble for an
    unknown name, or a disambiguation when the text matches more than one spell.
    Every mutator repaints the widget's own cells before returning, so a
    caller never needs to trigger a separate render pass.
    """

    WIDTH = 46

    def __init__(self, spells):
        Widget.__init__(self)
        self.spells = spells        # {name: factory}
        self.text = ""
        self.phase = 'name'
        self.spell_name = None
        self.message = ""
        self.done = False
        self.result = None

    def type_char(self, ch):
        self.text += ch
        self._changed()
        self.repaint()

    def backspace(self):
        self.text = self.text[:-1]
        self._changed()
        self.repaint()

    def needs_direction(self):
        """Whether the resolved spell wants a direction before it is cast.

        A projectile spell does (the direction phase follows); a self-cast
        spell like *glo* does not, and ``run_cast`` finishes the moment its name
        resolves. Defaults to True for a spell that does not say (see
        ``spell.Spell.NEEDS_DIRECTION``).
        """
        return getattr(self.spells[self.spell_name], 'NEEDS_DIRECTION', True)

    def resolve(self):
        """Match the typed text against the spell names; advance on a hit.

        Matching is by in-order subsequence, not contiguous substring, so a few
        characteristic letters name a spell: "bal" finds *fireball* and "lit"
        finds *lightning*. On a unique match the spell name is set; a spell that
        wants a direction moves to the direction phase, a self-cast one stays in
        the name phase for ``run_cast`` to finish at once. Otherwise sets
        ``message``, stays in the name phase, and returns None.
        """
        typed = self.text.strip().lower()
        matches = [name for name in self.spells
                   if typed and _subsequence(typed, name)]
        if len(matches) == 1:
            self.spell_name = matches[0]
            self.message = ""
            if self.needs_direction():
                self.phase = 'direction'
            self._changed()
            self.repaint()
            return matches[0]
        if not matches:
            self.message = ('You mumble "%s" -- nothing happens.'
                            % (self.text.strip() or '...'))
        else:
            self.message = "Did you mean %s?" % " or ".join(sorted(matches))
        self._changed()
        self.repaint()
        return None

    # -- sizing/drawing ------------------------------------------------

    def preferred_shape(self):
        # title, blank, prompt, message, hint
        return 5, self.WIDTH

    def _shape_changed(self):
        self.repaint()

    def repaint(self):
        """Redraw the whole widget from current state (phase, text/message)."""
        self.clear(fg=FG, bg=BG)
        self.write(0, 0, "Cast a spell", fg=TITLE_FG)
        if self.phase == 'name':
            self.write(2, 0, "Spell: " + self.text + "_")
            hint = "Type a name, Enter to cast, Esc to cancel"
        else:
            self.write(2, 0, "%s -- which way?" % self.spell_name.capitalize())
            hint = "Press an arrow key  (Esc to cancel)"
        if self.message:
            self.write(3, 0, self.message, fg=HINT_FG)
        self.write(self.nrows - 1, 0, hint, fg=HINT_FG)


def run_cast(session, prompt):
    """Key handling for a CastWidget; returns ``(spell_name, arrow_key)`` or None.

    Name phase: printable characters build the name, Backspace edits, Enter
    resolves it. A self-cast spell (no direction) finishes right there with a
    None direction; a projectile spell advances to the direction phase, where
    the next arrow key finishes. Escape cancels at either phase. Key releases
    and unknown keys are ignored.
    """
    while True:
        event = session.get()
        if not isinstance(event, KeyPress):
            continue
        if event.key == 'Escape':
            return None
        if prompt.phase == 'name':
            if event.key in ('Enter', 'Return'):
                if prompt.resolve() is not None and not prompt.needs_direction():
                    return (prompt.spell_name, None)
            elif event.key == 'Backspace':
                prompt.backspace()
            elif event.text and len(event.text) == 1 and event.text.isprintable():
                prompt.type_char(event.text)
        elif event.key in ARROWS:
            return (prompt.spell_name, event.key)
