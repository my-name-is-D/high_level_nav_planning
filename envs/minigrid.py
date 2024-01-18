import numpy as np

class GridWorldEnv():
    
    def __init__(self, rooms:np.ndarray, actions:dict):
        self.state_mapping = {}
        self.rooms = rooms
        actions = {'LEFT':0,'RIGHT':1, 'UP':2, 'DOWN':3, 'STAY':4}
        self.possible_actions = {k.upper(): v for k, v in actions.items()}

        self.unknown_rooms = np.ones_like(rooms) -2
    
    def step(self,a, prev_p):
        #self.state = np.dot(self.B[:,:,a], self.state)
        next_pos = self.next_p_given_a_known_env(prev_p, a)
        
        # obs = utils.sample(np.dot(self.A, self.state))

        #obs = rooms[next_pos[0], next_pos[1]]
        self.update_rooms_obs(next_pos)
        obs = self.get_ob_given_p(next_pos)
        self.update_states(next_pos, obs)
        return obs, next_pos
    
    def hypo_step(self,a, prev_p):
        #self.state = np.dot(self.B[:,:,a], self.state)
        next_pos = self.next_p_given_a_known_env(prev_p, a)
        obs = self.get_ob_given_p(next_pos)
        return obs, next_pos

    def reset(self, pose):
        self.state_mapping = {}
        self.unknown_rooms = np.ones_like(self.rooms) -2
        # obs = rooms[pose[0], pose[1]]
        self.update_rooms_obs(pose)
        obs = self.get_ob_given_p(pose)
        self.update_states(pose, obs)
        
        # obs = utils.sample(np.dot(self.A, self.state))
        return obs
    
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
        return self.unknown_rooms[pose[0], pose[1]]
        
    def get_state(self,pose):
        ''' get state given pose'''
        return self.state_mapping[pose]['state']

    def next_p_given_a_known_env(self, prev_position, action):
        row, col = prev_position
        action_key = list(filter(lambda x: self.possible_actions[x] == action, self.possible_actions))[0] 

        #it's probably: actions = {'UP':2, 'RIGHT':1, 'DOWN':3, 'LEFT':0, 'STAY':4} , but let's stay safe
        if action_key == "LEFT" and 0 < col:
            col -= 1
        elif action_key == "RIGHT" and col < self.rooms.shape[1] - 1:
            col += 1
        elif action_key == "UP" and 0 < row:
            row -= 1
        elif action_key == "DOWN" and row < self.rooms.shape[0] - 1:
            row += 1

        return (row,col)

    def get_next_possible_motions(self, position:tuple)->list:
        row, col = position
        step_possible_actions = []
        for action_key, action in self.possible_actions.items():
            if (
            action_key == "STAY" or
            (action_key == "LEFT" and col > 0) or
            (action_key == "RIGHT" and col < self.rooms.shape[1] - 1) or
            (action_key == "UP" and row > 0) or
            (action_key == "DOWN" and row < self.rooms.shape[0] - 1)):
                step_possible_actions.append(action)
            
        return step_possible_actions
