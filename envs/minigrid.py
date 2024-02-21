import numpy as np
# from visualisation_tools import create_custom_cmap
from envs.modules import get_l2_distance, astar

class GridWorldEnv():
    
    def __init__(self, room_choice:str, actions:dict, max_steps:int = False, goal:int=-1, **kwargs):
        self.state_mapping = {}
        self.step_count = 0
        self.goal_ob = goal
        self.max_steps = max_steps
        self.curr_loc = (0,0)
        self.rooms, self.rooms_colour_map = setup_grid(room_choice)
        self.possible_actions = {k.upper(): v for k, v in actions.items()}
        self.unknown_rooms = np.ones_like(self.rooms) -2
    
    def step(self,a, prev_p=None):
        if prev_p is None:
            prev_p = self.curr_loc
        self.step_count +=1
        reward = 0
        #self.state = np.dot(self.B[:,:,a], self.state)
        next_pos = self.next_p_given_a_known_env(prev_p, a)
        self.curr_loc = next_pos
        # obs = utils.sample(np.dot(self.A, self.state))
        #obs = rooms[next_pos[0], next_pos[1]]
        # self.update_rooms_obs(next_pos) #that was for test purposes. usefull when generating own ob based on 'new or not'
        colour_ob = self.get_ob_given_p(next_pos)
        self.update_states(next_pos, colour_ob)
        if colour_ob == self.goal_ob:
            reward = self._reward()
        done = False
        
        return [colour_ob, next_pos] , reward,  done, {}
    
    def hypo_step(self,a, prev_p):
        #self.state = np.dot(self.B[:,:,a], self.state)
        next_pos = self.next_p_given_a_known_env(prev_p, a)
        obs = self.get_ob_given_p(next_pos)
        return obs, next_pos

    def reset(self, pose=None):
        if pose is None:
            pose = (0,0)
        self.state_mapping = {}
        self.step_count = 0
        self.curr_loc = pose
        self.unknown_rooms = np.ones_like(self.rooms) -2
        # obs = rooms[pose[0], pose[1]]
        # self.update_rooms_obs(pose)
        obs = self.get_ob_given_p(pose)
        self.update_states(pose, obs)
        
        # obs = utils.sample(np.dot(self.A, self.state))
        return [obs, pose], {}
    
    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        if self.max_steps:
            return 1 - 0.9 * (self.step_count / self.max_steps)
        else:
            return 1
        

    def get_short_term_goal(self, input=None):
        if input is None:
            start_pose = self.curr_loc
        elif isinstance(input, dict):
            start_pose = input['pose_pred']
        else:
            raise ValueError('get_short_term_goal:Input type not recognised ' + str(type(input)))

        output = [0,0,0]
        goal_poses_row, goal_poses_col = self.get_goal_position(self.goal_ob)
        all_relative_dists = []
        for r, c in zip(goal_poses_row,goal_poses_col):
            relative_dist = get_l2_distance(r, start_pose[0], c, start_pose[1])
            all_relative_dists.append(relative_dist)
        if len(all_relative_dists) == 0:
            #Goal not in env
            return None
        closest_goal_idx = np.argmin(all_relative_dists)
        goal_row = goal_poses_row[closest_goal_idx]
        goal_col = goal_poses_col[closest_goal_idx]
        path = astar(self.rooms, start_pose, (goal_row, goal_col))
        if path == None:
            return None
        output[0] = int((0%360.)/5.) #angle
        output[1] = len(path) -1 #step dist
        output[2] = path #gt path #NB: might not be the unique best path
        return output

    def get_goal_position(self, goal):
        #Get the goal pose as x, y or row, column
        goal_poses = np.where(self.rooms == goal)
        return goal_poses[0], goal_poses[1]

    def update_states(self,pose, ob):
        """ create state if needed """
        if pose not in self.state_mapping.keys():
            self.state_mapping[pose] = {'state' : len(self.state_mapping) , 'ob': ob}
    
    def update_rooms_obs(self,pose):
        if self.unknown_rooms[pose[0], pose[1]] < 0:
            new_ob = self.unknown_rooms.max()+1
            self.unknown_rooms[pose[0], pose[1]] = new_ob

            real_ob = self.rooms[pose[0], pose[1]]
            self.unknown_rooms[self.rooms == real_ob] = new_ob
        
    def get_ob_given_p(self,pose):
        return self.rooms[pose[0], pose[1]]
        
    def get_state(self,pose):
        ''' get state given pose'''
        return self.state_mapping[pose]['state']

    def next_p_given_a_known_env(self, prev_position, action):
        row, col = prev_position
        action_key = list(filter(lambda x: self.possible_actions[x] == action, self.possible_actions))[0] 

        #it's probably: actions = {'UP':2, 'RIGHT':1, 'DOWN':3, 'LEFT':0, 'STAY':4} , but let's stay safe
        if action_key == "LEFT" and 0 < col and self.rooms[row][col-1] >= 0:
            col -= 1
        elif action_key == "RIGHT" and col < self.rooms.shape[1] - 1 and self.rooms[row][col+1] >= 0:
            col += 1
        elif action_key == "UP" and 0 < row and self.rooms[row-1][col] >= 0:
            row -= 1
        elif action_key == "DOWN" and row < self.rooms.shape[0] - 1 and self.rooms[row+1][col] >= 0:
            row += 1

        return (row,col)

    def get_next_possible_motions(self, position:tuple, no_stay=False)->list:
        row, col = position
        step_possible_actions = []
        for action_key, action in self.possible_actions.items():
            if (
            action_key == "STAY" or
            (action_key == "LEFT" and col > 0 and self.rooms[row][col-1] >= 0) or
            (action_key == "RIGHT" and col < self.rooms.shape[1] - 1 and self.rooms[row][col+1] >= 0) or
            (action_key == "UP" and row > 0 and self.rooms[row-1][col] >= 0) or
            (action_key == "DOWN" and row < self.rooms.shape[0] - 1 and self.rooms[row+1][col] >= 0)):
                step_possible_actions.append(action)
        if no_stay and 'STAY' in self.possible_actions.keys():
            step_possible_actions.remove(self.possible_actions['STAY'])
        return step_possible_actions
    
    def get_possible_motions(self):
        step_possible_actions = []
        for action_key, action in self.possible_actions.items():
            if action_key != 'STAY':
                step_possible_actions.append(action)
        return step_possible_actions
    
    def define_perfect_B(self):
        """ The perfect B is defined as B[next_state, prev_state, action]"""
        #perfect B for this room config
        desired_state_mapping = {idx: coord for idx, coord in enumerate(((i, j) \
                for i in range(self.rooms.shape[0]) for j in range(self.rooms.shape[1]) if self.rooms[i, j] != -1))}
        P = {}
        dim = self.rooms.shape
        for state_index, xy_coordinates in desired_state_mapping.items():
            P[state_index] = {a : [] for a in range(len(self.possible_actions))}
            for action in self.possible_actions.values():
                pose = self.next_p_given_a_known_env(xy_coordinates, action)
                #print('action', action, 'state coordinates', state_index, xy_coordinates, 'next pose', pose)
                next_state_idx = next(key for key, value in desired_state_mapping.items() if value == pose)
                P[state_index][action] = next_state_idx


        num_states = len(desired_state_mapping)
        B = np.zeros([num_states, num_states, len(self.possible_actions)])
        # print(B.shape)
        for s in range(num_states):
            # print('s', s, perfect_state_mapping[s])
            for a in range(len(self.possible_actions)):
                ns = int(P[s][a])
                # print('ps', s, 'a', a, 'ns',ns)
                B[ns, s, a] = 1
        return B, desired_state_mapping
    
    

    
    
def setup_grid(room_choice:str = 'grid_3x3'):
    
    room_choice = room_choice.lower()
    if room_choice == 'grid_3x3_alias': #3x3 rooms, 1 ob per room - WT ALIAS
        rooms = np.array(
            [
                [0, 0, 1],
                [2, 0, 4],
                [3, 3, 3],
            ]
        )
    
    elif  room_choice == "grid_3x3": #3x3 rooms, 1 ob per room - NO ALIAS

        rooms = np.array(
            [
                [0, 1, 2],
                [5, 4, 3],
                [6, 7, 8],
            ]
        ) 
    elif  room_choice == "grid_3x3_test": #3x3 rooms, 1 ob per room - NO ALIAS

        rooms = np.array(
            [
                [0, 1, 2],
                [4, 5, 3],
                [8, 7, 6],
            ]
        ) 

    elif  room_choice == "grid_3x4_alias": #3x4 rooms, 1 ob per room - WT ALIAS

        rooms = np.array(
            [
                [0, 2, 1, 4],
                [2, 0, 1, 3],
                [3, 0, 3, 0],
            ]
        ) #3x4 rooms, 1 ob per room
    
    elif  room_choice == "grid_4x4_alias": #4x4 rooms, 1 ob per room - WT ALIAS

        rooms = np.array(
            [
                [0, 0, 1, 4],
                [2, 1, 1, 3],
                [3, 5, 3, 0],
                [1, 0, 4, 0],
            ]
        ) #4x4 rooms, 1 ob per room

    elif  room_choice == "grid_4x4": #4x4 rooms, 1 ob per room - NO ALIAS

        rooms = np.array(
            [
                [0, 1, 2, 3],
                [7, 6, 5, 4],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ]
        )
    
    elif  room_choice == "grid_t_maze_alias": #T-maze with alias

        rooms = np.array(
            [
                [0, 1, 2, 1, 0],
                [-1, -1, 3, -1, -1],
                [-1, -1, 3, -1, -1],
                [-1, -1, 3, -1, -1],
            ]
        )

    elif  room_choice == "grid_t_maze": #T-maze with alias - NO ALIAS

        rooms = np.array(
                [
                    [0, 1, 2, 3, 7],
                    [-1, -1, 4, -1, -1],
                    [-1, -1, 5, -1, -1],
                    [-1, -1, 6, -1, -1],
                ]
            )
        
    elif  room_choice == "grid_donut": #Squared all around - NO ALIAS

        rooms = np.array(
                [
                    [0, 1, 2, 3, 4],
                    [5, -1, -1, -1, 13],
                    [6, -1, -1, -1, 12],
                    [7, 8, 9, 10, 0],
                ]
            )
    else:
        raise ValueError("Room_choice "+ str(room_choice) +" is an invalid choice")
    
    custom_colors = (
        np.array(
            [
                [255, 255, 255],#white
                [255, 0, 0],#red
                [0, 255, 0], #green
                [50,50, 255], #bluish 
                [112, 39, 195], #purple
                [255, 255, 0], #yellow
                [100, 100, 100], #grey
                [115, 60, 60], #brown
                [255, 0, 255], #flash pink
                [80, 145,80], #kaki
                [201,132,226], #pink
                [75,182,220], #turquoise
                [255,153,51], #orange
                [255,204,229], #light pink
                [153,153,0], #ugly brown 
                [229,255,204], #light green
                [204,204,255],#light purple
                [0, 153,153], #dark turquoise
            ]
        )
        / 256
    )

    cmap = create_custom_cmap(custom_colors[:rooms.max()+1])
    return rooms, cmap

def create_custom_cmap(custom_colors):
    from matplotlib import colors
    return colors.ListedColormap(custom_colors[:]) #,  alpha=None)
