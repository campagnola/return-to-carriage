from enum import Enum
import operator as op
import numpy as np



class DialogueMatrix(object):
    DEFAULT_MARKER = 3

    def __init__(self, name, parent):
        self.parent = parent
        self.name = name
        self.uid_generator = 0

        self.marker_uid = None
        self.default_marker()
        # use the marker_uid to initialize a specific starting dialogue node for the next time the matrix.run() is executed with target=None.
        # e.g. Player has a fight with spouse, so the next time they meet, the conversation starts out sullen.
        
        self.nodes = {}
        self.actions = {}
        self.routes = {}
        self.conditions = {}

    def generate_uid(self):
        self.uid_generator += 1
        return self.uid_generator

    def add_node(self, node):
        self.nodes[node.uid] = node
        return(node.uid)

    def remove_node(self, uid):
        self.nodes.remove(uid)

    def add_route(self, route):
        self.routes[route.uid] = route
        return(route.uid)

    def remove_route(self, route_uid):
        self.routes.remove(route_uid)

    def add_action(self, method, args):
        uid = self.generate_uid()
        self.actions[uid] = [method, args]
        return(uid)

    def remove_action(self, action_uid):
        self.actions.remove(action_uid)

    def add_condition(self, subj, obj, oper, neg=False):
        uid = self.generate_uid()
        self.conditions[uid] = RouteCondition(subj, obj, oper, neg)
        return(uid)

    def remove_condition(self, condition_uid):
        self.conditions.remove(condition_uid)

    def set_marker(self, uid):
        self.marker_uid = uid

    def default_marker(self):
        self.marker_uid = self.DEFAULT_MARKER

    def get_node(self, target=None):
        if target is None:
            target = int(self.marker_uid)
            print('--- using marker target, {0}'.format(self.marker_uid))

        if target in self.nodes.keys():
            return self.nodes[target]
        else:
            print('\nwtf: run_matrix() did not find target in nodes.keys()\n')
            return None

    def get_action(self, uid):
        return self.actions[uid]


class RouteCondition(object):
    def __init__(self, subj, obj, oper, neg=False):
        # subj = the thing doing the operation
        self.subj = subj
        # obj = the thing being tested by the operation
        self.obj = obj
        # oper = the operand
        self.oper = oper
        # neg = invert the bool output
        self.neg = neg

        # e.g.   fishtank (subj), fish (obj), operator.contains (oper), True (invert the output)
        #        "The fishtank contains fish: False." 
        
    def test_condition(self):
        b = self.oper(self.subj, self.obj)
        if self.neg:
            b = not b
        return b


class DialogueRoute(object):
    def __init__(self, matrix, uid=None, name='', target_uid=None):
        self.matrix = matrix

        if uid is None:
            uid = self.matrix.generate_uid()
        self.uid = uid
        self.name = name
        self.target_uid = target_uid

        self.conditions = []
        self.output_actions = []    
    
    def set_target(self, target_uid):
        self.target_uid = target_uid

    def add_action(self, action_uid):
        self.actions.append(action_uid)

    def remove_action(self, action_uid):
        self.actions.pop(action_uid)

    def add_condition(self, condition_uid):
        self.conditions.append(condition_uid)

    def remove_condition(self, condition_uid):
        self.conditions.pop(condition_uid)

    def test_all_conditions(self):
        b = True
        for condition in self.conditions:
            if condition.test_condition() == False:
                return False
        return True


class DialogueNode(object):
    class NodeType(Enum):
        TEXT = 1        #   the node chain endpoint. this is the only node that holds text data
        MULTI = 2       #   multiple choice group (generally, the routes on the output of a MULTI node point only to TEXT nodes, 
                        #           which is the standard choice for Player dialogue)
        VALUE = 3       #   picks output route based on a single value, evenly scaled across all routes
                        #   e.g.    if the value in question is the relationship with an NPC, hypothetically 7.8 (of 10) and there
                        #           are 4 total routes, then 7.5+ meets the condition for the first route, 5.0 to 7.4 = 2nd route, etc
        COMPLEX = 4     #   complex conditional type is for checking one or more bools and/or comparison operations
                        #           the conditions live in the routes. Check each route in list order - the first one to pass all 
                        #           conditions is the winner
                        #           e.g.   if (magic_fire_arrow in player.inventory)  and  (player.health > 0.90):  do the thing
        RANDOM = 5      #   a group of routes, selected randomly. these routes almost always point toward TEXT nodes
        
    def __init__(self, matrix, node_type, parent_node=None, uid=None, name='', text='', character_uid=0):
        self.matrix = matrix
        self.node_type = self.decode_node_type(node_type)
        self.parent_node = parent_node

        if uid is None:
            uid = self.matrix.generate_uid()
        self.uid = uid
        self.name = name

        self.text = text
        self.character_uid = character_uid

        self.input_actions = []
        self.routes = []

        self.value_var = None

    def decode_node_type(self, node_type):
        if node_type in   ['text',      'TEXT',     't', 'T', 1, '1', 'txt', 'TXT', self.NodeType.TEXT]:
            return self.NodeType.TEXT
        elif node_type in ['multi',     'MULTI',    'm', 'M', 2, '2', self.NodeType.MULTI]:
            return self.NodeType.MULTI
        elif node_type in ['value',     'VALUE',    'v', 'V', 3, '3', self.NodeType.VALUE]:
            return self.NodeType.VALUE
        elif node_type in ['complex',   'COMPLEX',  'c', 'C', 4, '4', self.NodeType.COMPLEX]:
            return self.NodeType.COMPLEX
        elif node_type in ['random',    'RANDOM',   'r', 'R', 5, '5', self.NodeType.RANDOM]:
            return self.NodeType.RANDOM

    def set_uid(self, uid):
        self.uid = uid

    def set_name(self, name):
        self.name = name

    def add_input_action(self, uid):
        self.input_actions.append(uid)

    def remove_input_action(self, uid):
        self.input_actions.pop(uid)
        
    def add_route(self, uid):
        self.routes.append(uid)

    def remove_route(self, uid):
        self.routes.pop(uid)

    def set_character_uid(self, character_uid):
        self.character_uid = character_uid


if __name__ == '__main__':
    class Player:
        character_uid = 0
        name = "Sobel"
    player = Player()

    class NPC:
        character_uid = 1
        name = "Shopkeep"
        relationships = {player: 5}  # 0=bad 10=good
        inventory = {   
                        'apples': 5,
                        'oranges': 2,
                        'whiffle balls': 900        }
    npc = NPC()

    class Conversation:
        def __init__(self, player, npc):
            self.player = player
            self.npc = npc
            self.matrix = DialogueMatrix('test_convo', self)

        def get_node(self, target=None):
            node = self.matrix.get_node(target)
            return node
        
        def challenge_node_types(self, node):
            i = None
            route = None
            s = ''

            if node.node_type == node.NodeType.TEXT:
                self.print_node_text(node.text, node.character_uid)
                if len(node.routes) == 1:
                    route = node.routes[0].uid
                elif len(node.routes) != 1:
                    print('wtf. Conversation.challenge_node_types NodeType.TEXT')
                
            elif node.node_type == node.NodeType.MULTI:
                #gather all the text from the downstream TEXT nodes
                i = 0
                for route in node.routes:
                    i += 1
                    s += '{0}. '.format(int(i))
                    s += self.matrix.nodes[self.matrix.routes[route.uid].target_uid].text
                    s += '\n'
                self.print_node_text(s, node.character_uid)
                r = ''
                while True:
                    r = raw_input('\n>')
                    if r.isdigit() == True:
                        if len(node.routes) >= int(r) >= 1:
                            break
                    # add some more escape measures
                route = node.routes[int(r) - 1].uid
                
            elif node.node_type == node.NodeType.VALUE:
                v = float(node.value_var)
                v = abs(v - 1.0)
                num_of_routes = float(len(node.routes))
                i = v / num_of_routes
                route = node.routes[i].uid

            elif node.node_type == node.NodeType.COMPLEX:
                route = self.complex_node_find_first_valid_route(node.routes)

            elif node.node_type == node.NodeType.RANDOM:
                i = np.random.randint(0, len(node.routes))
                route = node.routes[i]

            return route

        def run_node(self, node):
            self.execute_actions(node.input_actions)
            
            i = None
            s = ''

            route = self.challenge_node_types(node)
            
            if route is None:
                print('wtf. run_node(node) is not returning a route uid')
            else:
                self.execute_route(route)

        def execute_actions(self, actions=[]):
            if len(actions) > 0:
                for item in actions:
                    method, args = self.matrix.actions[item]
                    print('--- Executing action->', method, args)

        def print_node_text(self, text, character_uid=0):
            if character_uid == 0:
                print(text, '\n')
            else:
                print('{:>40}'.format(text), '\n')     
                  
        def complex_node_find_first_valid_route(self, routes=[]):
            if len(routes) == 0:
                return None

            for route in routes:
                b = True
                for condition in route.conditions:
                    if self.matrix.conditions[condition].test_condition() == False:
                        b = False
                        break
                if b == True:
                    return route.uid

        def check_all_conditions(self, conditions=[]):
            if len(conditions) == 0:
                return True
            for condition in conditions:
                if condition.test_condition() == False:
                    return False
            return True

        def execute_route(self, route):
            self.execute_actions(self.matrix.routes[route].output_actions)
            node = self.get_node(self.matrix.routes[route].target_uid)
            print('--- Running next node: {0}'.format(self.matrix.routes[route].target_uid))
            self.run_node(node)



    conversation = Conversation(player, npc)

    def run_test_conversation():
        node = conversation.get_node()
        print('--- Running default node')
        conversation.run_node(node)

    def build_test_conversation():
        text_node = conversation.matrix.add_node(DialogueNode(conversation.matrix, 'text', text='What can I do for you today?', character_uid=1))
        route = conversation.matrix.add_route(DialogueRoute(conversation.matrix, target_uid=text_node))
        random_node = conversation.matrix.add_node(DialogueNode(conversation.matrix, 'random', character_uid=1))
        conversation.matrix.nodes[random_node].add_route(route)
    
"""
      DialogueNode( matrix, node_type, parent_node=None, uid=None, name='', text='', character_uid=0 )
      DialogueRoute( matrix, uid=None, name='', target_uid=None )
      RouteCondition( subj, obj, oper, neg=False )

      How to build a conversation:
      1. Optimally, draw out the entire network of coversation pathways, including node types and descriptions per node cluster
          (by 'node cluster', i'm referring to any grouping of nodes that ends with TEXT type nodes, and commonly beginning with VALUE
          type nodes, though technically, a cluster can start with any node type. "Cluster" is what was formerly "Dialogue Column" - 
          the widest level of grouping before the top level DialogueMatrix  )
      2. As the destination uids need to be know in order to finish the data of a node, it makes the most practical sense to build the 
          conversation backwards, from right to left.
      3. We can add code that automatically provides the "parent_uid" to the nodes once their target nodes have been entered manually.
          (and as it is right now, "parent_uid" is yet unused)
      4. For complex conversations, this process will take a very high level of menial grunt work, ergo be on the lookout for how to 
          streamline or automate said conversation construction
      5. A step-by-step (just to get the feel)
        - Start by writing out all possible script text lines for one particular cluster (maybe in the middle of the conversation)
        - Come up with a convention for short-term / local variable naming so that you can create around 10 or 20 TEXT nodes in a single 
            cluster. (this would include all variations, such as Player-NPC relationship branching (VALUE), etc)
        - Now create the prior routes for each of the TEXT nodes (the upstream routes). Supply the uid from the TEXT nodes (the return value 
            from conversation.matrix.add_node() ) to the 'target_uid' arguments when defining the upstream Routes.
            Should look like this:
>              
>       text_node = conversation.matrix.add_node(DialogueNode(conversation.matrix, 'text', text='What can I do for you today?', character_uid=1))
>       route = conversation.matrix.add_route(DialogueRoute(conversation.matrix, target_uid=text_node))

        - Next, add another set of nodes, using node_type='random'.
        - Append the 'route' variables to the random_node 'routes' list attribute

>       random_node = conversation.matrix.add_node(DialogueNode(conversation.matrix, 'random', character_uid=1))
>       conversation.matrix.nodes[random_node].add_route(route)



            
    







"""

