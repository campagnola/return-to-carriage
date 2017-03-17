from asciimatics.screen import Screen
from asciimatics.event import KeyboardEvent, MouseEvent
import time
import numpy as np


def make_maze(h, w):
    maze = np.zeros((h, w), dtype=np.uint)
    
    # outer walls
    maze[0] = 1
    maze[-1] = 1
    maze[:,0] = 1
    maze[:,-1] = 1

    return maze


def print_maze(screen, maze):
    for i,row in enumerate(maze):
        row = ''.join(np.array(list(".#"))[row])
        screen.print_at(row, 0, i, Screen.COLOUR_WHITE)

def event_loop(screen):
    pos = [1, 1]
    wall = ' '
    maze = make_maze(screen.height, screen.width)
    print_maze(screen, maze)
    
    
    while True:
        refresh = False
        while True:
            ev = screen.get_event()
            if ev is None:
                break
            refresh = True
            if isinstance(ev, KeyboardEvent):
                key = ev.key_code
                msg = str(key)
                screen.print_at(' ', *pos, Screen.COLOUR_WHITE)
                if key == screen.KEY_LEFT:
                    pos[0] = max(1, pos[0]-1)
                if key == screen.KEY_RIGHT:
                    pos[0] = min(screen.width-2, pos[0]+1)
                if key == screen.KEY_UP:
                    pos[1] = max(1, pos[1]-1)
                if key == screen.KEY_DOWN:
                    pos[1] = min(screen.height-2, pos[1]+1)
                screen.print_at(' ', *pos, Screen.COLOUR_WHITE, Screen.A_REVERSE)
            elif isinstance(ev, MouseEvent):
                msg = "%d %d %s" % (ev.x, ev.y, ev.buttons)
            else:
                msg = "Unknown event type: %s" % str(ev)
        if refresh:
            screen.print_at(msg, 5, screen.height-1, Screen.COLOUR_GREEN)
            screen.refresh()
        time.sleep(0.02)

Screen.wrapper(event_loop)


