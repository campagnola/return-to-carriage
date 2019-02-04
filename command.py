import re
from collections import OrderedDict
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
from PyQt4 import QtCore
from errors import ActionError
from input import MenuInputHandler


class CommandInterpreter(object):
    def __init__(self, scene):
        self.scene = scene
        self.player = self.scene.player
        self.partial = None
        self._menu_items = {}

    def __call__(self, command):
        """Interpret commands in the form "verb arg1 arg2 ...".

        Each verb corresponds to a method of this class.
        """

        args = re.split(r'\s+', command)
        if len(args) == 0:
            self.scene.console.write("I beg your pardon?")
            return

        verb, args = args[0], args[1:]
        fn = getattr(self, verb, None)
        if fn is None or not callable(fn):
            if self.partial is not None:
                # attempt to restart a partial command
                command = self.partial + command
                self.partial = None
                try:
                    self(command)
                finally:
                    self._menu_items = {}
            else:
                self.scene.console.write('You lost me at "%s"' % verb)
            return

        try:
            self.scene.console.write("\n> %s" % command)
            response = fn(args)
            if response is not None:
                self.scene.console.write(response)
        except:
            self.scene.console.write('Your attempt has torn the spacetime continuum.\n'
                                     'With any luck the rift will pass unnoticed, but you resolve to be more cautious in the future (if there is one).')
            raise

    def take(self, args):
        items = self.scene.items_at(self.player.position)
        if len(args) == 0:
            if len(items) == 0:
                return "You take, but nothing gives."
            if len(items) > 1:
                # maybe offer up a menu
                self.scene.console.write("You feel the urge to hoard:")
                return self._partial_item_menu('take', items) 
            take_items = items
        else:
            # see if we can figure out which item(s) are wanted.
            take_items = []
            for name in args:
                possible_items = self._items_matching(name, items)
                if len(possible_items) == 0:
                    if len(name) == 1:
                        return 'You double-check, but %s is definitely not on the menu.' % name
                    else:
                        return 'You thought there was a "%s" around here, but perhaps that was just your imagination.' % name
                if len(possible_items) > 1:
                    return 'Despite the variety of "%s" to choose from, indecision paralyzes you and you resolve to try again later.' % name
                take_items.append(possible_items[0])
        
        for item in take_items:
            try:
                self.player.take(item)
            except ActionError as exc:
                return exc.reason

    def drop(self, args):
        items = [i for i in list(self.player.inventory.values()) if i is not None]
        if len(args) == 0:
            self.scene.console.write("Drop which item?")
            return self._partial_item_menu('drop', items)

        drop_items = []
        for name in args:
            possible_items = self._items_matching(name, items)
            if len(possible_items) == 0:
                if len(name) == 1:
                    return 'You double-check, but %s is definitely not on the menu.' % name
                else:
                    return 'You thought there was a "%s" around here, but perhaps that was just your imagination.' % name
            if len(possible_items) > 1:
                return 'Despite the variety of "%s" to choose from, indecision paralyzes you and you resolve to try again later.' % name
            drop_items.append(possible_items[0])

        for item in drop_items:
            try:
                self.player.drop(item)
            except ActionError as exc:
                return exc.reason

    def _items_matching(self, search, items):
        match = []
        if len(search) == 1:
            if search in self._menu_items:
                item = self._menu_items[search]
                if item in items:
                    match.append(item)
        else:
            for item in items:
                if search in item.description:
                    match.append(item)
        return match

    def _partial_item_menu(self, partial, items):
        self.partial = partial + " "
        self._menu_items = OrderedDict()
        for i, item in enumerate(items):
            letter = chr(97 + i)
            self.scene.console.write("  %s: %s" % (letter, item.description))
            self._menu_items[letter] = item

        if not self.scene.command_mode:
            self._menu_input_handler = MenuInputHandler(self)
            self._menu_input_handler.activate()

