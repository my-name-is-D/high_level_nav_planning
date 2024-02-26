import numpy as np
from visualisation_tools import B_to_ideal_B


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


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


def agent_B_match_ideal_B_v2(agent_B, perfect_B, agent_state_mapping, desired_state_mapping, actions, tolerance_margin = 0.3):
    """Check if the values == 1 in perfect_B are filled with values relatively close at tolerance level"""
    room_valid_state_agent= { k:v for k, v in agent_state_mapping.items() if k in desired_state_mapping.values() }

    if len(room_valid_state_agent) < len(desired_state_mapping):
        return False
    rearranged_B = B_to_ideal_B(agent_B, actions,desired_state_mapping, room_valid_state_agent)
    rearranged_B = rearranged_B[:len(desired_state_mapping),:len(desired_state_mapping),:]
    if rearranged_B.shape != perfect_B.shape:
        raise 'rearranged_B.shape should match B.shape'
    non_zero_mask = perfect_B > 0
    matching_indices = np.where(non_zero_mask & (np.abs(perfect_B - rearranged_B) <= tolerance_margin))
    match_result = np.array_equal(np.array(matching_indices) ,np.array(np.where(non_zero_mask)))
    
    return match_result


def discretize(dist):
    dist_limits = [0.25, 3, 10]
    dist_bin_size = [0.05, 0.25, 1.]
    if dist < dist_limits[0]:
        ddist = int(dist/dist_bin_size[0])
    elif dist < dist_limits[1]:
        ddist = int((dist - dist_limits[0])/dist_bin_size[1]) + \
            int(dist_limits[0]/dist_bin_size[0])
    elif dist < dist_limits[2]:
        ddist = int((dist - dist_limits[1])/dist_bin_size[2]) + \
            int(dist_limits[0]/dist_bin_size[0]) + \
            int((dist_limits[1] - dist_limits[0])/dist_bin_size[1])
    else:
        ddist = int(dist_limits[0]/dist_bin_size[0]) + \
            int((dist_limits[1] - dist_limits[0])/dist_bin_size[1]) + \
            int((dist_limits[2] - dist_limits[1])/dist_bin_size[2])
    return ddist


def heuristic(curr_pos, goal):
    return abs(curr_pos[0] - goal[0]) + abs(curr_pos[1] - goal[1])

def astar(env, start, goal):
    if env[start[0]][start[1]] == -1 or env[goal[0]][goal[1]] == -1:
        return None  # Start or goal position is blocked
   
    rows, cols = len(env), len(env[0])
    visited = set()
    pq = [(0, start, [])]  # (f_score, position, path)
    
    while pq:
        pq.sort(reverse=True)  # Sort descending based on f_score
        f_score, curr_pos, path = pq.pop()
        if curr_pos == goal:
            return path + [curr_pos]
        
        visited.add(curr_pos)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = curr_pos[0] + dr, curr_pos[1] + dc
            if 0 <= r < rows and 0 <= c < cols and env[r][c] != -1 and (r, c) not in visited:  
                new_path = path + [curr_pos]
                new_f_score = len(new_path) + heuristic((r, c), goal)
                pq.append((new_f_score, (r, c), new_path))
                visited.add((r, c))
    
    return None  # Goal not reachable

        
    #     visited.add(curr_pos)
    #     for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
    #         r, c = curr_pos[0] + dr, curr_pos[1] + dc
    #         if 0 <= r < rows and 0 <= c < cols and env[r][c] != 1 and (r, c) not in visited:
    #             new_f_score = len(path) + 1 + heuristic((r, c), goal)
    #             pq.append((new_f_score, (r, c), path + [curr_pos]))
    #             visited.add((r, c))
    
    # return None  # Goal not reachable