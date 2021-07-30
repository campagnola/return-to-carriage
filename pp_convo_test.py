import random


def start():
    """NPC greeting"""
    disp = npc.disposition_to(player)
    if disp > 1:
        npc.say(random.choice(["Hey buddy.", "'sup, friend.", "Yo!"]))
    elif -1 < disp < 1:
        npc.say(random.choice(["Welcome!", "How can I help?"]))
    else:
        npc.say("Make it quick.")
    return choose_item

def choose_item():
    """Player select item of interest"""
    choice = choose("Red potion, please", "10 of your finest arrows.", "You know what I want.", None)
    conv.request, next_node = [
        (('red potion', 1), make_offer),
        (('arrow', 10), make_offer),
        (None, offer_cat),
        (None, leave),
    ][choice]
    if choice == 2:
        npc.adjust_disposition(player, -0.5)

    return next_node

def make_offer():
    """Make offer to player"""
    if npc.disposition_to(player) > 1:
        conv.discount = 0.5
        npc.say(f"I'll give ya a deal. {npc.offer_price(conv)} gourd.")
    else:
        conv.discount = 1.0
        total_price = npc.offer_price(conv)
        item_price = total_price / conv.request[1]
        npc.say(f"{conv.request[0].capitalize()}s are hot. {item_price} each comes out to {total_price} gourd.")

    return consider_offer

def consider_offer():
    """Accept or reject offer"""
    choice = choose("I accept.", "No way!")
    if choice == 0:
        if player.money >= npc.offer_price(conv):
            return accept_offer
        else:
            scene.narrate("You dig around in our purse as your error dawns on you.")
            player.say("Wait. I changed my mind.")
            npc.adjust_disposition(player, -0.2)
            return reject_offer
    else:
        return reject_offer

def accept_offer():
    """Offer accepted"""
    price = npc.offer_price(conv)
    npc.say(f"It's a bargain at {price} gourd!")
    npc.adjust_disposition(player, 0.4)
    player.inventory.append(conv.request)
    conv.request = None
    player.money -= price
    return choose_item

def reject_offer():
    """Offer refused"""
    npc.say("Come back when you're serious.")
    npc.adjust_disposition(player, -0.2)
    return choose_item

def offer_cat():
    npc.say("*sigh* Not again. C'mon man, I'm all outta cats.")
    if not player.saw_cats:
        player.say("Next time, then.")
        return choose_item

    player.say("No you're not. Let's see 'em.")
    npc.adjust_disposition(player, -0.4)
    npc.say("What the hell'd you do with the last two?")
    player.say("None of your business. Got any siamese?")
    npc.say("...")
    npc.say("You're a creepy dude.")
    npc.say("150 for the siamese.")
    conv.discount = 1.0
    conv.request = ('siamese cat', 1)
    return consider_offer

def leave():
    scene.narrate("You wander the streets until finding yourself back at the shop.")
    return start


if __name__ == '__main__':

    class Player:
        def __init__(self):
            self.inventory = []
            self.money = 200
            self.saw_cats = True

        def say(self, msg):
            print("\nPlayer:", msg)


    class NPC:
        def __init__(self):
            self._disp = {}
        
        def disposition_to(self, player):
            return self._disp.get(player, 0)

        def adjust_disposition(self, player, amount):
            self._disp.setdefault(player, 0)
            self._disp[player] += amount

        def say(self, msg):
            print("\n         NPC:", msg)
        
        def offer_price(self, conv):
            item, qty = conv.request
            item_price = {
                'red potion': 15,
                'arrow': 6,
                'siamese cat': 150,
            }[item]

            return int(item_price * qty * conv.discount)


    class Scene:
        def narrate(self, text):
            print("\n  **  ", text)

    class ConversationState:
        pass


    player = Player()
    npc = NPC()
    scene = Scene()
    conv = ConversationState()


    def choose(*choices):
        print("\nSay:")
        for i,ch in enumerate(choices):
            print(f"    {i}) {ch or '[ leave ]'}")
        response = input()
        try:
            choice = int(response)
            assert choice < len(choices)
            player.say(choices[choice])
            return choice
        except Exception:
            print("You chose .. poorly:", response)
            return choose(*choices)


    node = start
    while True:
        node = node()
