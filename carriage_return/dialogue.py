from enum import Enum
import numpy as np
import time
import msvcrt
import random


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

        self.request = None
        self.discount = 1.0

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

    def __init__(self, convo, group, uid=None, node_type=1, name='', text='', opt=[], char_uid=0, next=[], val=None, delay=2.0, input_actions=[], output_actions={}):
        self.convo = convo
        self.group = group

        self.node_type = node_type

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
        self.delay = delay
        self.input_actions = input_actions
        self.output_actions = output_actions

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
        n = None
        time.sleep(self.delay)

        # execute input actions
        for s in self.input_actions:
            exec(s)


        if self.node_type == self.TEXT:
            self.print_text(self.text, self.character_uid)
            n = 0


        elif self.node_type == self.MULTI:
            self.print_text(self.text, self.character_uid)
            a = None
            while a not in self.valid_input:
                a = msvcrt.getch().decode('utf-8')
                if a == '\x1b':
                    break
            i = int(a)-1
            print('\n{0}\n'.format(self.opt[i]))
            n = int(i)


        elif self.node_type == self.VALUE:
            i = 0.0
            j = 0
            if self.val is None:
                print('self.val is None')
                return

            # sample the attribute. val indices [0] and [1] == obj, 'attr'
            i = getattr(*self.val[:2])
            # if self.val only contains [0] and [1], then run the automated behavior where the value (assumed range 0.0 to 1.0) is scaled across the output option indices 
            if len(self.val) == 2:
                j = int(i*len(self.opt))
            # but if there is a 3rd list member, [2], then run through each val[2] value until i > val[2][n]
            # e.g. if getattr returned 0.88 to i, and val[2] == [0.9, 0.6], then we will be conditionally selecting the 2nd index (1) because 0.88 falls between the 1st group
            # (0.9 to 1.0) and the 3rd group (0.0 to 0.6) 
            elif len(self.val) == 3:
                for v in self.val[2]:
                    if i >= v:
                        break
                    j += 1

            if len(self.opt):
                #yes, I'm making an assumption here. 
                if len(self.opt[j]):
                    self.print_text(self.opt[j], self.character_uid)
            n = int(j)


        elif self.node_type == self.COMPLEX:
            pass


        elif self.node_type == self.RANDOM:
            # all this shit basically says: if self.opt contains options, then select a random index associated with a self.opt index and print that text
            if len(self.opt):
                i = random.randint(0, len(self.opt)-1)
                self.print_text(self.opt[i], self.character_uid)
                # then, if there are as many 'next' members as there are 'opt' members, that means each 'opt' gets its own 'next'
                if len(self.next) == len(self.opt):
                    n = i
                # if the 'next' and 'opt' lists are of different lengths, then that means self.next[] contains only 1 member, 
                # and all options share this same 'next'   (next[0]) 
                else:
                    n = 0
            # else if there are no options supplied, then we are just selecting a random 'next' output, and we're not worried about printing text from this node. 
            # checking here if len(self.next) returns > 0 is unnecessary, but it is somewhat more futureproof 
            elif len(self.next):
                n = random.randint(0, len(self.next)-1)


        # execute output actions
        if n in self.output_actions.keys():
            for s in self.output_actions[n]:
                exec(s)

        # pass the ball
        self.convo.run(self.next[n])




if __name__ == '__main__':
    class NPC():
        def __init__(self, r):
            self.relationship = r
            self.item_price = {
                                'red potion': 15,
                                'arrow': 6,
                                'siamese cat': 150,
                            }
    npc = NPC(0.7)

    class Player():
        def __init__(self):
            self.money = 200
            self.inventory = []
            self.saw_cats  = True
    player = Player()

    c = Conversation('test')
    c.cast.append(player)
    c.cast.append(npc)

    c.marker = 2
    c.nodes[2] = Node(c, 0, uid=2,  node_type=Node.VALUE,
                                    next=[3, 4, 5],
                                    val=(npc,'relationship')    )

    c.nodes[3] = Node(c, 0, uid=3,  node_type=Node.RANDOM,
                                    opt=['Hey buddy.', "'sup, friend?", 'Yo!'],
                                    next=[6],
                                    char_uid=1    )
    c.nodes[4] = Node(c, 0, uid=4,  node_type=Node.RANDOM,
                                    opt=['Welcome', "How can I help?"],
                                    next=[6],
                                    char_uid=1     )
    c.nodes[5] = Node(c, 0, uid=5,  node_type=Node.RANDOM,
                                    opt=['Make it quick.'],
                                    next=[6],
                                    char_uid=1     )

    c.nodes[6] = Node(c, 0, uid=6,  node_type=Node.MULTI,
                                    opt=['Red potion, please', '10 of your finest arrows', 'You know what I want', '*exit' ],
                                    next=[7, 7, 8, 9],
                                    output_actions={    0: ['self.convo.request = ("red potion", 1)'],
                                                        1: ['self.convo.request = ("arrows", 10)'], 
                                                        2: ['self.convo.cast[1].relationship -= 0.25'] }   )

    c.nodes[7] = Node(c, 0, uid=7,  node_type=Node.VALUE,
                                    next=[8,9],
                                    val=(npc, 'relationship', [0.667]),
                                    output_actions={    0: ["self.convo.discount = 0.5"], 
                                                        1: ["self.convo.discount = 1.0"]   }   )
    c.nodes[8] = Node(c, 0, uid=8,  node_type=Node.TEXT,
                                    text='',
                                    next=[10],
                                    input_actions=['self.text="I\'ll give ya a deal.\\n{0} gourd.".format(self.convo.cast[1].item_price[self.convo.request[0]] * self.convo.request[1] * self.convo.discount)'],
                                    char_uid=1      )

    c.nodes[9] = Node(c, 0, uid=9,  node_type=Node.TEXT,
                                    char_uid=1,
                                    text='',
                                    input_actions=["self.text='{0}s are hott! {1} each comes out to {2} gourd'.format( self.convo.request[0], \
                                                                                            npc.item_price[self.convo.request[0]], \
                                                                                            npc.item_price[self.convo.request[0]] * self.convo.request[1] )" ] ,
                                        next=[10]       )
    c.nodes[10] = Node(c, 0, uid=10,    node_type=Node.MULTI,
                                        opt=['I accept.', 'No way!'],
                                        next=[11, 12]   )

    c.nodes[11] = Node(c, 0, uid=11,    node_type=Node.VALUE,
                                        next=[13, 14],
                                        input_actions=["self.val=(player, 'money', [ npc.item_price[self.convo.request[0]] * self.convo.request[1] ]) "] )

    c.nodes[12] = Node(c, 0, uid=12,    node_type=Node.TEXT,
                                        text='',
                                        next=[6],
                                        char_uid=1,
                                        input_actions=["self.text='It\'s a bargain at {0} bucks.'.format(npc.item_price[c.request[0]] * c.request[1])" ],
                                        output_actions={0:['self.convo.cast[1].relationship += 0.2', \
                                                            'self.convo.cast[0].money -= self.convo.cast[1].item_price[c.request[0]] * self.convo.request[1])', \
                                                            'self.convo.cast[0].inventory.append([self.convo.request[0], self.convo.request[1]])']  }     )

    c.nodes[13] = Node(c, 0, uid=13,    node_type=Node.TEXT,
                                        text="Come back when you're serious, kid.",
                                        next=[6],
                                        char_uid=1,
                                        output_actions={0: ['c.cast[1].relationship -= 0.2']}     )

























"""
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



"""