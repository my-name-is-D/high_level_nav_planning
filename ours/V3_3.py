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
class Ours_V3_3():
    def __init__(self, num_obs=2, num_states=2, observations=[0,(0,0)], learning_rate_pB=3.0, actions= {'Left':0}, utility_term=True) -> None:
        self.agent_state_mapping = {}
        self.pose_mapping = []
        self.possible_actions = actions
        self.agent = None
        self.initialisation(num_obs=num_obs, num_states=num_states, observations=observations, learning_rate_pB=learning_rate_pB, utility_term= utility_term)

    def initialisation(self,num_obs=2, num_states=2, observations=[0,(0,0)], learning_rate_pB=3.0, utility_term=True):
        """ Create agent and initialise it with env first given observations """
        pose  = observations[1] #start pose in map
        o = observations[0]
        self.pose_mapping.append(pose)
        p_idx = self.pose_mapping.index(pose)

        #INITIALISE AGENT
        B_agent = create_B_matrix(num_states,len(self.possible_actions))
        pB = utils.to_obj_array(B_agent)
        A_agent = create_A_matrix(num_obs,num_states,2)
        pA = utils.dirichlet_like(A_agent, scale = 1)
        self.agent = Agent(A = A_agent, pA=pA, B = B_agent , pB = pB,policy_len= len(self.possible_actions), lr_pB=learning_rate_pB, 
                    inference_algo="VANILLA", save_belief_hist = True, use_utility=utility_term, \
                    action_selection="stochastic",use_param_info_gain=True)
        
        self.agent.qs[0] = utils.onehot(0, num_states)
        self.agent.qs_hist.append(self.agent.qs)
        self.update_A_with_data([o,p_idx],0)
        return self.agent
    
    def update_preference(self, C):
        """TODO:Will be used with an observation to add in C"""
        if C is not None:
            if not isinstance(C, np.ndarray):
                raise TypeError(
                    'C vector must be a numpy array'
                )
            self.agent.C = utils.to_obj_array(C)

            assert len(self.agent.C) == self.agent.num_modalities, f"Check C vector: number of sub-arrays must be equal to number of observation modalities: {self.num_modalities}"

            for modality, c_m in enumerate(self.agent.C):
                assert c_m.shape[0] == self.agent.num_obs[modality], f"Check C vector: number of rows of C vector for modality {modality} should be equal to {self.num_obs[modality]}"
        else:
            self.agent.C = self.agent._construct_C_prior()

    def update_agent_state_mapping(self, pose:tuple, ob:int, state_belief:list=None)-> dict:
        """ Dictionnary to keep track of believes and associated obs, usefull for testing purposes mainly"""
        if state_belief is None:
            state = -1
        else:
            state = np.argmax(state_belief)
        self.agent_state_mapping[pose] = {'state' : state , 'ob': ob}
        return self.agent_state_mapping

    def get_current_belief(self):
        return self.agent.qs
    
    def infer_action(self, next_possible_actions=None):
        q_pi, G = self.agent.infer_policies()
        action = self.agent.sample_action()
        return int(action)

    def get_agent_state_mapping(self, x=None,a=None, agent_pose=None)->dict:
        return self.agent_state_mapping
    
    def get_B(self):
        return self.agent.B[0]
    
    def get_n_states(self):
        return len(self.agent_state_mapping)
    #==== Update A and B ====#
    
    def update_A_with_data(self,obs:list, state:int)->np.ndarray:
        """Given obs and state, update A entry """
        A = self.agent.A
        for dim,ob in enumerate(obs):
            A[dim][:,state] = 0
            A[dim][ob,state] = 1
        self.agent.A = A
        return A

    def update_A_dim_given_obs_3(self, obs:list,null_proba:list=[True]) -> np.ndarray:
        ''' 
        Verify if the observations are new and fit into the current A shape.
        If not, increase A shape in observation (n row) only.
        '''
        A = self.agent.A
        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=A)
        
        # Calculate the maximum dimension increase needed across all modalities
        dim_add = [int(max(0,obs[m] + 1 - num_obs[m])) for m in range(num_modalities)]

        # Update matrices size
        for m in range(num_modalities):
            A[m] = update_A_matrix_size(A[m], add_ob=dim_add[m], null_proba=null_proba[m])
            if self.agent.pA is not None:
                self.agent.pA[m] = utils.dirichlet_like(utils.to_obj_array(A[m]), scale=1)[0]
        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=A)
        self.agent.num_obs = num_obs
        self.agent.A = A
        return A

    def update_B_dim_given_A(self)-> np.ndarray:
        """ knowing A dimension, update B state dimension to match"""
        B = self.agent.B
        add_dim = self.agent.A[0].shape[1]-B[0].shape[1]
        if add_dim > 0: 
            #increase B dim
            B = update_B_matrix_size(B, add= add_dim)
            self.agent.pB = update_B_matrix_size(self.agent.pB, add= add_dim)
            self.agent.qs[0] = np.append(self.agent.qs[0],[0]*add_dim)
        
        self.agent.num_states = [B[0].shape[0]]
        self.agent.B = B
        return B
    
    def update_A_dim_given_pose(self, pose_idx:int,null_proba:bool=True) -> np.ndarray:
        ''' 
        Verify if the observations are new and fit into the current A shape.
        If not, increase A shape and associate those observations with the newest state generated.
        '''
        A = self.agent.A
        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=A)
        
        # Calculate the maximum dimension increase needed across all modalities
        dim_add = int(max(0,pose_idx + 1 - num_obs[1]))
        # Update matrices size
        #and associate new observations with the newest state generated
        if dim_add > 0:
            A[0] = update_A_matrix_size(A[0], add_state=dim_add, null_proba=null_proba)
            A[1] = update_A_matrix_size(A[1], add_ob=dim_add, add_state=dim_add, null_proba=null_proba)
            #we search the first fully null or normed column (thus no link between state -> ob) #THIS IS MAINLY FOR SECURITY
            columns_wthout_data = np.sort(np.append(np.where(np.all(A[1] == 1/A[1].shape[0], axis=0))[0], np.where(np.all(A[1] == 0, axis=0))[0]))
            A[1][:, columns_wthout_data[0]] = 0
            A[1][pose_idx, columns_wthout_data[0]] = 1
            if self.agent.pA is not None:
                self.agent.pA = utils.dirichlet_like(utils.to_obj_array(A), scale=1)
            
        
        self.agent.num_obs[1] = A[1].shape[0]
        self.agent.num_states = [A[1].shape[1]]
        self.agent.A = A
        return A
    
    def update_C_dim(self):
        if self.agent.C is not None:
            num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=self.agent.A) 
            print(num_modalities, num_states, num_factors)
            for m in range(num_modalities):
                print(m, num_obs[m])
                if self.agent.C[m].shape[0] < num_obs[m]:
                    self.agent.C[m] = np.append(self.agent.C[m], [0]*(num_obs[m]- self.agent.C[m].shape[0]))
                    
    #==== update Believes ====#
    def update_A_belief(self,obs:list)->None:
        #UPDATE A given all observations
        #IDENTIFY WHERE THE AGENT BELIEVES TO BE
        Qs = self.agent.infer_states(obs, distr_obs=False)[0] 
        self.agent.update_A(obs)
        self.agent.update_A(obs) #twice to increase effect (not mandatory)
        
    def update_believes_v2(self, Qs:list, action:int, obs:list)-> None:
        #UPDATE B
        if len(self.agent.qs_hist)+1 > 1:#secutity check
            if len(self.agent.qs_hist[-1][0]) < len(Qs[0]):
                self.agent.qs_hist[-1][0] = np.append(self.agent.qs_hist[-1][0],[0]*(len(self.agent.qs_hist[-1][0])-len(Qs[0])))
            self.update_B(Qs, self.agent.qs_hist[-1], action, lr_pB = 10) 
            #2 WAYS TRANSITION UPDATE (only if T to diff state)
            if np.argmax(self.agent.qs_hist[-1][0]) != np.argmax(Qs[0]):
                a_inv = reverse_action(self.possible_actions, action)
                self.update_B(self.agent.qs_hist[-1], Qs, a_inv, lr_pB = 5)
        
        self.update_A_belief(obs)
    

    def add_ghost_node_v3(self,p_idx:int, possible_next_actions:list)-> None:
        ''' 
        For each new pose observation, add a ghost state and update the estimated transition and observation for that ghost state.
        '''
        pose = self.pose_mapping[p_idx]
        for action in self.possible_actions.values():
            if action not in possible_next_actions: #this mean this action is not deemed possible
                self.update_B(self.agent.qs, self.agent.qs, action, lr_pB = 10)
            else:
                n_pose = next_p_given_a(pose, self.possible_actions, action)
                if n_pose not in self.pose_mapping:
                    self.pose_mapping.append(n_pose)
                    p_idx = self.pose_mapping.index(n_pose)
                    print('a',action,'n pose', n_pose, p_idx)
                    self.update_A_dim_given_pose(p_idx, null_proba=False) #we only update pose ob and assign a state to this ob
                    self.update_B_dim_given_A()
                    hypo_qs = self.infer_states_no_history([p_idx], partial_ob=1)
                    self.update_B(hypo_qs, self.agent.qs, action, lr_pB = 3) 
                    self.update_agent_state_mapping(n_pose, -1, hypo_qs[0])
                    # a_inv = reverse_action(self.possible_actions, action)
                    # self.update_B(self.agent.qs, hypo_qs, a_inv, lr_pB = 1)
        

    #==== PYMDP modified methods ====#

    def update_B(self,qs:np.ndarray, qs_prev:np.ndarray, action:int, lr_pB:int=None)-> np.ndarray:
        """
        Update posterior beliefs about Dirichlet parameters that parameterise the transition likelihood 
        
        Parameters
        -----------
        qs_prev: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
            Marginal posterior beliefs over hidden states at previous timepoint.

        Returns
        -----------
        qB: ``numpy.ndarray`` of dtype object
            Posterior Dirichlet parameters over transition model (same shape as ``B``), after having updated it with state beliefs and actions.
        """
        
        if lr_pB is None:
            lr_pB = self.agent.lr_pB

        qB = update_state_likelihood_dirichlet(
            self.agent.pB,
            self.agent.B,
            [action],
            qs,
            qs_prev,
            lr_pB,
            self.agent.factors_to_learn
        )

        self.agent.pB = qB # set new prior to posterior
        self.agent.B = utils.norm_dist_obj_arr(qB)  # take expected value of posterior Dirichlet parameters to calculate posterior over B array
        return qB

    def infer_states_no_history(self, observation, distr_obs = False, partial_ob:int=None)->np.ndarray:
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.
        Don't save the inferred state in qs_hist or agent.qs
        Parameters
        ----------
        observation: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores the index of the discrete
            observation for modality ``m``.

        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """

        observation = tuple(observation) if not distr_obs else observation

        if not hasattr(self.agent, "qs"):
            self.agent.reset()

        if self.agent.inference_algo == "VANILLA":
            if self.agent.action is not None:
                empirical_prior = control.get_expected_states(
                    self.agent.qs, self.agent.B, self.agent.action.reshape(1, -1) #type: ignore
                )[0]
            else:
                empirical_prior = self.agent.D
            qs = update_posterior_states(
            self.agent.A,
            observation,
            empirical_prior,
            partial_ob,
            **self.agent.inference_params
            )
        elif self.agent.inference_algo == "MMP":

            self.agent.prev_obs.append(observation)
            if len(self.agent.prev_obs) > self.agent.inference_horizon:
                latest_obs = self.agent.prev_obs[-self.agent.inference_horizon:]
                latest_actions = self.agent.prev_actions[-(self.agent.inference_horizon-1):]
            else:
                latest_obs = self.agent.prev_obs
                latest_actions = self.agent.prev_actions

            qs, F = inference.update_posterior_states_full(
                self.agent.A,
                self.agent.B,
                latest_obs,
                self.agent.policies, 
                latest_actions, 
                prior = self.agent.latest_belief, 
                policy_sep_prior = self.agent.edge_handling_params['policy_sep_prior'],
                **self.agent.inference_params
            )

        return qs

    #==== BELIEVES PROCESS UPDATE ====#
    def agent_step_update(self, action, observations = [0,(0,0)], possible_next_actions:list=[0,1,2,3]):
        ob = observations[0]
        pose = observations[1]
        if pose not in self.pose_mapping:
            self.pose_mapping.append(pose)
        p_idx = self.pose_mapping.index(pose)

        # prev_state_size = agent.num_states[0]
        #3. UPDATE A AND B DIM WITH THOSE DATA
        self.update_A_dim_given_obs_3([ob,p_idx], null_proba=[False,False])
        self.update_B_dim_given_A()
        # new_state_size = agent.num_states[0]

        #4. UPDATE BELIEVES GIVEN OBS
        Qs = self.infer_states_no_history([ob,p_idx], distr_obs=False)
        print('prior on believed state', Qs[0].round(3))
        
        
        #4.5 UPDATE A AND B WITH THOSE BELIEVES
        self.update_believes_v2(Qs, action, [ob,p_idx])
        self.update_agent_state_mapping(pose, ob, self.agent.qs[0])

        print('after updates believed Qs:',self.agent.qs[0].round(3))
        #ADD KNOWLEDGE WALL T OR GHOST NODES
        self.add_ghost_node_v3(p_idx, possible_next_actions)
        self.update_C_dim()














