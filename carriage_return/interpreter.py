"""Game-side action layer for player commands.

CommandInterpreter is the single home for the take/read/drop triage shared by
two input paths: the typed console commands (``interp("take")``) and the
gameplay shortcuts (``interp.take([])`` from the t/r/d keys). It decides "no
items / one item / show a menu", opens modal dialogs when a choice is needed,
and delegates the actual mechanics to the player. It is fully game-side — no
rendering library is imported here; output goes to ``scene.log`` and dialogs
open through the game-side ``dialogs`` package.
"""
import re

from . import dialogs, spell
from .errors import ActionError


class CommandInterpreter(object):
    """Interprets player commands ("take", "drop a torch", ...) against a Scene.

    The scene supplies the player, the message log, and world queries; menus
    open through the game-side dialogs package and complete on their own
    thread.
    """
    def __init__(self, scene):
        self.scene = scene

    @property
    def player(self):
        return self.scene.player

    def __call__(self, command):
        """Interpret a typed command in the form "verb arg1 arg2 ...".

        Each verb corresponds to a method of this class.
        """
        args = re.split(r'\s+', command.strip())
        verb, args = args[0], args[1:]
        fn = getattr(self, verb, None)
        if fn is None or not callable(fn) or verb.startswith('_'):
            self.scene.write('You lost me at "%s"' % verb)
            return

        try:
            self.scene.write("\n> %s" % command)
            fn(args)
        except:
            self.scene.write('Your attempt has torn the spacetime continuum.\n'
                             'With any luck the rift will pass unnoticed, but you resolve to be more cautious in the future (if there is one).')
            raise

    # -- take ---------------------------------------------------------------

    def take(self, args):
        """Take item(s) from the ground under the player.

        No items: a message. One item (or an unambiguous args match): take it.
        Several items and no args: open a multi-select menu; the take happens
        when the menu is accepted, on the dialog thread.
        """
        items = self.scene.items_at(self.player.location.slot)
        if len(args) == 0:
            if len(items) == 0:
                self.scene.write("You take, but nothing gives.")
            elif len(items) == 1:
                self._take(items)
            else:
                session = dialogs.open_menu(self.scene, "Take which items?", items,
                                            multi_select=True)
                session.finished.connect(
                    lambda s: s.result and self._take(s.result))
        else:
            take_items = self._match_args(args, items)
            if take_items is not None:
                self._take(take_items)

    def _take(self, items):
        """Move *items* into the player's inventory, reporting each outcome.

        Runs on the gameplay or dialog thread (the sole game-state mutator at
        the time); only ActionError is reported-and-continued.
        """
        for item in items:
            try:
                self.player.take(item)
            except ActionError as exc:
                self.scene.write(exc.reason)
            else:
                self.scene.write("Taken: %s." % item.description)

    # -- drop ---------------------------------------------------------------

    def drop(self, args):
        """Drop item(s) from the player's inventory.

        No args: open a multi-select menu over the inventory (the drop happens
        when it is accepted). With args: drop the unambiguous matches.
        """
        items = list(self.player.inventory.all_entities())
        if len(args) == 0:
            if len(items) == 0:
                self.scene.write("You have nothing to drop.")
                return
            session = dialogs.open_menu(self.scene, "Drop which items?", items,
                                        multi_select=True)
            session.finished.connect(
                lambda s: s.result and self._drop(s.result))
        else:
            drop_items = self._match_args(args, items)
            if drop_items is not None:
                self._drop(drop_items)

    def _drop(self, items):
        for item in items:
            try:
                self.player.drop(item)
            except ActionError as exc:
                self.scene.write(exc.reason)

    # -- cast ---------------------------------------------------------------

    def cast(self, args):
        """Open the spell-casting prompt: type a spell name, then a direction.

        The prompt captures all input while open (a modal DialogSession); when
        it completes with a ``(spell, arrow)`` pair the spell is created at the
        player's feet, flying in the chosen direction. A self-cast spell (glo)
        completes with ``arrow`` None and acts on the player alone. Cancelled
        with Escape -- or resolved to no spell -- it does nothing. The cast
        happens on the dialog thread, the sole game-state mutator while the
        prompt is up.
        """
        session = dialogs.open_cast(self.scene, spell.SPELLS)
        session.finished.connect(
            lambda s: s.result is not None and self._cast(*s.result))
        return session

    def _cast(self, spell_name, arrow):
        """Create *spell_name* at the player, flying toward *arrow*.

        *arrow* is an arrow-key name from the prompt, mapped to a maze vector by
        ``spell.DIRECTIONS``; it is None for a self-cast spell, which ignores
        the direction. The spell places itself and begins animating.
        """
        maze, pos = self.player.location.place
        direction = spell.DIRECTIONS[arrow] if arrow is not None else None
        spell.SPELLS[spell_name](self.scene, maze, pos, direction)

    # -- read ---------------------------------------------------------------

    def read(self, args):
        """Read a readable item from the inventory or the ground.

        None: a message. One readable: read it. Several: open a single-select
        menu, then read the chosen item on the dialog thread.
        """
        readables = [i for i in self.player.inventory.all_entities() if i.readable]
        readables += [i for i in self.scene.items_at(self.player.location.slot)
                      if i.readable]
        if len(readables) == 0:
            self.scene.write("You have nothing to read.")
        elif len(readables) == 1:
            self.player.read(readables[0])
        else:
            session = dialogs.open_menu(self.scene, "Read which item?", readables)
            session.finished.connect(
                lambda s: s.result is not None and self.player.read(s.result))

    # -- helpers ------------------------------------------------------------

    def _match_args(self, args, items):
        """Resolve *args* (item-name fragments) against *items*.

        Returns the matched items, or None after writing a message when any
        name matches nothing or is ambiguous.
        """
        matched = []
        for name in args:
            possible = [item for item in items if name in item.description]
            if len(possible) == 0:
                if len(name) == 1:
                    self.scene.write('You double-check, but %s is definitely not on the menu.' % name)
                else:
                    self.scene.write('You thought there was a "%s" around here, but perhaps that was just your imagination.' % name)
                return None
            if len(possible) > 1:
                self.scene.write('Despite the variety of "%s" to choose from, indecision paralyzes you and you resolve to try again later.' % name)
                return None
            matched.append(possible[0])
        return matched
