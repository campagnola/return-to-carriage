from enum import Enum
import numpy as np
import time
import msvcrt



class Conversation(object):
    def __init__(self, name, parent=None):
        self.parent = parent
        self.name = name
        self.description = ''

        self.marker = None

        # list of all characters in this conversation [(character_uid, 'char name')]
        self.cast = []

        self.groups = {}
        self.nodes = {}

        self.uid_generator = 0

    def generate_uid(self):
        self.uid_generator += 1
        return self.uid_generator

    def run(self, uid=None):
        if uid is None:
            uid = self.marker

        self.nodes[uid].run()


class Group(object):
    def __init__(self, convo, uid=None, name='', description='', ):
        self.convo = convo

        if uid is None:
            uid = self.convo.generate_uid()
        self.uid = uid
        self.name = name
        self.description = description

        self.nodes = []


class Node(object):
    TEXT = 1        #   
    MULTI = 2       #   multiple choice group  
    VALUE = 3       #   picks output route based on a single value, evenly scaled across all routes
    COMPLEX = 4     #   complex conditional type is for checking one or more bools and/or comparison operations
    RANDOM = 5      #   a group of routes, selected randomly. these routes almost always point toward TEXT nodes

    def __init__(self, convo, group, uid=None, node_type=1, name='', text='', opt=[], char_uid=0, next=[], val=None, delay=2.0):
        self.convo = convo
        self.group = group

        self.node_type = node_type
        self.delay = delay

        if uid is None:
            uid = self.convo.generate_uid()
        self.uid = uid
        self.name = name

        self.text = text
        self.opt = opt
        self.character_uid = char_uid

        self.next = []
        for n in next:
            self.next.append(n)

        self.val = val

        self.valid_input = []
        if self.node_type == self.MULTI and len(self.opt) > 0:
            for s in opt:
                self.valid_input.append(str(self.opt.index(s)+1))
                self.text += '{0}. {1}\n'.format((self.opt.index(s)+1), s)


    def print_text(self, text, character_uid=0):
        if character_uid == 0:
            print(text, '\n')
        else:
            print('{:>80}'.format(text), '\n')   

    def run(self):
        time.sleep(self.delay)

        if self.node_type == 1:
            self.print_text(self.text, self.character_uid)
            self.convo.run(self.next[0])
        elif self.node_type == 2:
            self.print_text(self.text, self.character_uid)
            a = None
            while a not in self.valid_input:
                a = msvcrt.getch().decode('utf-8')
                if a == '\x1b':
                    break
            i = int(a)-1
            print('\n{0}\n'.format(self.opt[i]))
            self.convo.run(self.next[i])

        elif self.node_type == 3:
            i = 0.0
            if self.val is not None:
                if isinstance(self.val, (int, float)):
                    pass
                elif isinstance(self.val, (tuple, list)):
                    try:
                        i = getattr(*self.val)
                    except:
                        print('trying to run node of type MULTI. no bueno')
            i = int(i*len(self.opt))
            if len(self.opt[i]):
                self.print_text(self.opt[i], self.character_uid)
            self.convo.run(self.next[i])



        elif self.node_type == 4:
            pass
        elif self.node_type == 5:
            pass
        else:
            print("wtf? didn't catch node_type in node.run()")



if __name__ == '__main__':
    class NPC():
        def __init__(self, r):
            self.relationship = r
    npc = NPC(0.7)

    c = Conversation('test')
    c.cast.append('Player')
    c.cast.append('Shopkeep')


    c.marker = 2
    c.nodes[2] = Node(c, 0, 2, name='nPlayerGreeting_01', text='Hello.', next=[4])

    c.nodes[4] = Node(c, 0,  4, name='nShopkeepGreeting_01', text='Hi, welcome!\n', char_uid=1, next=[5])
    c.nodes[5] = Node(c, 0,  5, name='nShopkeepGreeting_02', text='What can I help you with today?\n', char_uid=1, next=[7], delay=1.5)

    c.nodes[7] = Node(c, 0, 7,  node_type=Node.MULTI, name='nPlayerMulti_01', 
                                opt=['Wherz duh shitter?', 'I need some plutonium.', 'Have you seen my cat?', 'Nice shoes.'],
                                next=[8,9,10,11]    )

    c.nodes[8] = Node(c, 0, 8, text='Well since you asked SO politely, it is down the hall to the left.', char_uid=1, next=[12])
    c.nodes[9] = Node(c, 0, 9, text="Sorry sir, but you can't just walk into a store and buy plutonium.", char_uid=1, next=[7])
    c.nodes[10] = Node(c, 0, 10, text='Sorry. I have not. *burp*', char_uid=1, next=[7])
    c.nodes[11] = Node(c, 0, 11, text='Thanks. Wanna fuck?', char_uid=1, next=[13])
    c.nodes[12] = Node(c, 0, 12,    node_type=3,
                                    opt=["You gettin' shitty wit me boy?", '*You ignore the man and go relieve yourself.'],
                                    next=[14, 0],
                                    val=(npc,'relationship')    )
    c.nodes[13] = Node(c, 0, 13, text="I thought you'd never ask.", next=[0])
    c.nodes[14] = Node(c, 0, 14, text="Run along now. One sympathizes.", char_uid=1, next=[0])



