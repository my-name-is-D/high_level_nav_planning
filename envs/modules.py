import numpy as np


def reverse_action(actions:dict, action:int)->int:
    actions =  {k.upper(): v for k, v in actions.items()}
    action_key = list(filter(lambda x: actions[x] == action, actions))[0] 

    #IF MINIGRID ENV
    if 'DOWN' in actions.keys(): 
        if 'LEFT' in action_key: 
            reverse_action_key = 'RIGHT'
        elif 'RIGHT' in action_key: 
            reverse_action_key = 'LEFT'
        elif 'UP' in action_key: 
            reverse_action_key = 'DOWN'
        else:
            reverse_action_key = 'UP'

    reverse_action = actions[reverse_action_key]
    return reverse_action

def next_p_given_a(prev_position:tuple, actions:dict, action:int)->tuple:
    row, col = prev_position

    actions =  {k.upper(): v for k, v in actions.items()}
    action_key = list(filter(lambda x: actions[x] == action, actions))[0] 

    #IF MINIGRID ENV
    if 'DOWN' in actions.keys(): 
        #it's probably: actions = {'UP':2, 'RIGHT':1, 'DOWN':3, 'LEFT':0, 'STAY':4} , but let's stay safe
        if action_key == "LEFT" :
            col -= 1
        elif action_key == "RIGHT" :
            col += 1
        elif action_key == "UP" :
            row -= 1
        elif action_key == "DOWN" :
            row += 1
    
    return (row,col)