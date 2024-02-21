import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd

from ours.pymdp import utils
from ours.pymdp import control, inference
from ours.pymdp.agent import Agent
from envs.modules import reverse_action, next_p_given_a
from ours.modules import *

#==== INIT AGENT ====#
class Ours_V1():
    def __init__(self, num_obs=2, num_states=2, observations=[0,0], learning_rate_pB=3.0, actions= {'Left':0}) -> None:
        self.agent_state_mapping = {}
        self.pose_mapping = []
        self.possible_actions = actions
        self.agent = None
        self.prev_state_idx = -1
        self.initialisation(num_obs=num_obs, num_states=num_states, observations=observations, learning_rate_pB=learning_rate_pB)

    def initialisation(self,num_obs=2, num_states=2, observations=[0,0], learning_rate_pB=3.0):
        """ Create agent and initialise it with env first given observations """
        state_idx  = observations[1] #start pose in map
        ob = observations[0]

        #INITIALISE AGENT
        B_agent = create_B_matrix(num_states,len(self.possible_actions))
        A_agent = create_A_matrix([num_obs],[num_states],1)
        self.agent = Agent(A = A_agent, B = B_agent, policy_len= len(self.possible_actions), lr_pB=learning_rate_pB, 
                    inference_algo="VANILLA", save_belief_hist = True, use_utility=False, \
                    action_selection="stochastic",use_param_info_gain=True)
        
        self.update_A_with_data_v1(ob,state_idx)
        self.agent.infer_states([ob]) 
        self.prev_state_idx = state_idx
        self.update_agent_state_mapping(ob, 0)
        return self.agent
    
    def get_belief_over_states(self):
        return self.agent.qs
    
    def update_agent_state_mapping(self, ob:int, state_belief:list=None)-> dict:
        """ FULLY WRONG STUFF Dictionnary to keep track of believes and associated obs, usefull for testing purposes mainly"""
        if state_belief is None:
            state = -1
        else:
            state = np.argmax(state_belief)
        
        self.agent_state_mapping[state] = {'state' : state , 'ob': ob}
        return self.agent_state_mapping

    def infer_action(self, next_possible_actions=None):
        q_pi, G = self.agent.infer_policies()
        action = self.agent.sample_action()
        return int(action)
    
    def get_agent_state_mapping(self, x=None,a=None, agent_pose=None)->dict:
        return self.agent_state_mapping
    
    def get_B(self):
        return self.agent.B[0]
    def get_A(self):
        return self.agent.A
    
    def set_utility_term(self, utility_term:bool):
        self.use_utility = utility_term
        
    def get_n_states(self):
        return len(self.agent_state_mapping)
    
    def update_preference(self, obs:list):
        """given a list of observations we fill C with thos as preference. 
        If we have a partial preference over several observations, 
        then the given observation should be an integer < 0, the preference will be a null array 
        """
        if isinstance(obs, list):
            self.update_A_dim_given_obs_3(obs, null_proba=[False]*len(obs))

            C = self.agent._construct_C_prior()

            for modality, ob in enumerate(obs):
                if ob >= 0:
                    ob_processed = utils.process_observation(ob, 1, [self.agent.num_obs[modality]])
                    ob = utils.to_obj_array(ob_processed)
                else:
                    ob = utils.obj_array_zeros([self.agent.num_obs[modality]])
                obs[modality] = ob[0]
            C = np.array(obs, dtype=object)

            if not isinstance(C, np.ndarray):
                raise TypeError(
                    'C vector must be a numpy array'
                )
            self.agent.C = utils.to_obj_array(C)

            assert len(self.agent.C) == self.agent.num_modalities, f"Check C vector: number of sub-arrays must be equal to number of observation modalities: {agent.num_modalities}"

            for modality, c_m in enumerate(self.agent.C):
                assert c_m.shape[0] == self.agent.num_obs[modality], f"Check C vector: number of rows of C vector for modality {modality} should be equal to {agent.num_obs[modality]}"

    def goal_oriented_navigation(self, obs=None):
        self.update_preference(obs)
        self.agent.use_param_info_gain = False
        #self.agent.use_states_info_gain = False #Should we
        self.agent.use_utility = True

    def explo_oriented_navigation(self):
        self.agent.use_param_info_gain = True
        #self.agent.use_states_info_gain = True #Should we
        self.agent.use_utility = False

    #==== Update A and B ====#
    def update_C_dim(self):
        if self.agent.C is not None:
            num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=self.agent.A) 
            for m in range(num_modalities):
                if self.agent.C[m].shape[0] < num_obs[m]:
                    self.agent.C[m] = np.append(self.agent.C[m], [0]*(num_obs[m]- self.agent.C[m].shape[0]))
                    
    def update_A_with_data_v1(self, ob, state):
        A = self.agent.A
        if ob >= A[0].shape[0]:
            A = update_A_matrix_size(A,add_ob= ob-A[0].shape[0]+1)
        if state >= A[0].shape[1]:
            A = update_A_matrix_size(A,add_state= state-A[0].shape[1]+1)
        
        A[0][:,state] = 0
        A[0][ob,state] = 1
        self.agent.A = A
        return A

    def update_B_with_data(self, action, p_state, n_state, double_way=True):
        B = self.agent.B
        if n_state >= B[0].shape[0]: 
        #increase B dim
            B = update_B_matrix_size(B, add=1)
            self.agent.qs[0] = np.append(self.agent.qs[0],0)
        mean_value = B[0][:,p_state,action].mean()
        
        B[0][:,p_state,action] = np.maximum(0, B[0][:, p_state, action] - mean_value) #reduce proba of moving to another state
        
        B[0][n_state, p_state, action]  += mean_value + 0.5 #a static add is not quite satisfactory
        if n_state != p_state:
            B[0][p_state,p_state,action] += mean_value #except moving to same state
            if double_way:
                a_inv = action+1 if action%2 ==0 else action-1
                
                mean_value = B[0][:,n_state,a_inv].mean()/B[0].shape[0]
                
                B[0][:,n_state,a_inv] = np.maximum(0, B[0][:, n_state, a_inv] - mean_value)
                B[0][n_state,n_state,a_inv] += mean_value
                B[0][p_state, n_state, a_inv]  += mean_value + 0.25
            

        B = utils.norm_dist_obj_arr(B)
        self.agent.B = B
        return B
    
    #==== BELIEVES PROCESS UPDATE ====#
    def agent_step_update(self, action, observations = [0,0], ):
        next_state_idx  = observations[1] #start pose in map
        ob = observations[0]
        #4. UPDATE A AND B WITH THOSE DATA
        self.update_A_with_data_v1(ob, next_state_idx)
        self.update_B_with_data(int(action), self.prev_state_idx, next_state_idx)

      
        #4. UPDATE BELIEVES GIVEN OBS
        Qs = self.agent.infer_states([ob], distr_obs=True)[0]
        #print('action taken:',int(action),',ob:', ob, 'state', next_state_idx ,'prior on believed state', Qs.round(3))
        self.prev_state_idx = next_state_idx
        self.update_C_dim()
    














