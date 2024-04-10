import numpy as np
from copy import deepcopy


from ours.pymdp import utils
from ours.pymdp import control, inference
from ours.pymdp.learning import update_obs_likelihood_dirichlet
from ours.pymdp.agent import Agent
from envs.modules import reverse_action, next_p_given_a
from ours.modules import *
from ours.pymdp.maths import spm_dot

#==== INIT AGENT ====#
class Ours_V4_2(Agent):
    def __init__(self, num_obs=2, num_states=2, dim=2, observations=[0,(0,0)], lookahead=4, \
                 learning_rate_pB=3.0, actions= {'Left':0}, \
                 set_stationary_B=True, inference_algo= 'VANILLA') -> None:
        self.agent_state_mapping = {}
        self.pose_mapping = []
        self.possible_actions = actions
        self.preferred_ob = [-1,-1]
        self.set_stationary_B = set_stationary_B
        self.step_possible_actions = list(self.possible_actions.values())
        self.lookahead_distance = False #lookahead in number of consecutive steps
        self.simple_paths = True #L shaped paths, less computationally extensive than full coverage paths

        observations, agent_params = self.create_agent_params(num_obs=num_obs, num_states=num_states, observations=observations, \
                            learning_rate_pB=learning_rate_pB, dim=dim, lookahead=lookahead, inference_algo = inference_algo)
        super().__init__(**agent_params)
        self.initialisation(observations=observations)
    
    def create_agent_params(self,num_obs:int=2, num_states:int=2, observations:list=[0,(0,0)], 
                       learning_rate_pB:float=3.0, dim:int=2, lookahead:int=4,inference_algo:str='VANILLA'):
        ob = observations[0]
        p_idx = -1
        if dim > 1:
            #start pose in map
            if len(observations) < 2:
                observations.append((0,0))
            self.current_pose = observations[1]
            self.pose_mapping.append(observations[1])
            p_idx = self.pose_mapping.index(observations[1])
            observations[1] = p_idx
            
        else:
            self.current_pose = (0,0)
            self.pose_mapping.append(self.current_pose)
            p_idx = self.pose_mapping.index(self.current_pose)
        
        #INITIALISE AGENT PARAMS
        B_agent = create_B_matrix(num_states,len(self.possible_actions))
        if 'STAY' in self.possible_actions and self.set_stationary_B:
            B_agent = set_stationary(B_agent,self.possible_actions['STAY'])
        pB = utils.to_obj_array(B_agent)

        obs_dim = [np.max([num_obs, ob + 1])] + ([np.max([num_obs, p_idx + 1])] if dim > 1 else [])
        A_agent = create_A_matrix(obs_dim,[num_states]*dim,dim)
        pA = utils.dirichlet_like(A_agent, scale = 1)

        return observations, {
            'A': A_agent,
            'B': B_agent,
            'pA': pA,
            'pB': pB,
            'policy_len': lookahead,
            'inference_horizon': lookahead,
            'lr_pB': learning_rate_pB,
            'inference_algo': inference_algo,
            'save_belief_hist': True,
            'action_selection': "stochastic",
            'use_param_info_gain': False
        }

    def initialisation(self,observations:list=[0,(0,0)], linear_policies:bool=True, E=None):
        """ Initialise agent with first current observation and verify that all parameters 
        are adapted for continuous navigation.
        linear_policies(bool): 
        if False: we try all combinaison of actions (exponential n_action^policy_len  -wth policy_len==lookahead-) )
        if True: We create linear path reaching at a lookahead DISTANCE (not number of consecutive actions) or NUM STEPS
        we make it linear if no 'STAY' actions, else it's polynomial.
        It's not linear because of the STAY action that is irregular and set only at the end of a policy.

        NOTE:linear_policies=True is only tailored for num_factor==1 and len(num_control)==1 
        """
      
        if linear_policies:
            self.init_policies(E)
        self.reset(start_pose=self.pose_mapping[0])
        if self.edge_handling_params["use_BMA"] and hasattr(self, "q_pi_hist"): #This is not compatible with our way of moving
            del self.q_pi_hist
            
        self.inference_params_dict = {'MMP':
                    {'num_iter': 6, 'grad_descent': True, 'tau': 0.25},
                    'VANILLA':
                    {'num_iter': 3, 'dF': 1.0, 'dF_tol': 0.001}}

        self.update_A_with_data(observations,0)
        self.update_agent_state_mapping(self.current_pose, observations, 0)
        self.infer_states(observation = observations, distr_obs=False, partial_ob=None)
        return 
    
    def initialise_current_pose(self, observations:list):
        qs = self.get_belief_over_states()
        sorted_qs = np.sort(qs[0])
        
        #Not mandatory, but security (not tested without)
        if self.current_pose is None:
            threshold = 0.5
        else:
            threshold = 0.7

        #If we are sure of a state (independent of number of states), we don't have pose as ob and A allows for pose
        if sorted_qs[-1]-sorted_qs[-2] >= threshold and len(observations) < 2 and len(self.A) > 1:
            p_idx = np.argmax(self.A[1][:,np.argmax(qs[0])])
            self.current_pose = self.pose_mapping[p_idx]
            print('updating believed pose given certitude on state:', self.current_pose)
    
    def init_policies(self, E=None):
        policies = create_policies(self.policy_len, self.possible_actions, lookahead_distance=self.lookahead_distance,simple_paths= self.simple_paths)
        self.policies = policies
        assert all([len(self.num_controls) == policy.shape[1] for policy in self.policies]), "Number of control states is not consistent with policy dimensionalities"
        
        all_policies = np.vstack(self.policies)

        assert all([n_c >= max_action for (n_c, max_action) in zip(self.num_controls, list(np.max(all_policies, axis =0)+1))]), "Maximum number of actions is not consistent with `num_controls`"
        # Construct prior over policies (uniform if not specified) 
        if E is not None:
            if not isinstance(E, np.ndarray):
                raise TypeError(
                    'E vector must be a numpy array'
                )
            self.E = E
            assert len(self.E) == len(self.policies), f"Check E vector: length of E must be equal to number of policies: {len(self.policies)}"
        else:
            self.E = self._construct_E_prior()

    #==== GET METHODS ====#
    def get_agent_state_mapping(self, x=None,a=None, agent_pose=None)->dict:
        return self.agent_state_mapping
    
    def get_B(self):
        return self.B[0]
    
    def get_A(self):
        return self.A

    def get_n_states(self):
        return len(self.agent_state_mapping)
        
    def get_belief_over_states(self, Qs=None, n_step_past=0, verbose=False):
        """ 
        Extract a mean qs over policies if qs is in format qs[policy][timestep]
        It choses the current_qs by default unless a n_step_past is given. 
        (only valid when we have a number of steps < self.inference horizon, 
        else it's always qs_step).
        
        """
        if Qs is None:
            Qs = self.qs
            
        if len(Qs) == 1:
            my_qs = Qs
            
        else:
            
            qs_copy = [q[:self.policy_len + 1] for q in Qs] #In case we have policies of various length.
            current_qs_idx = self.qs_step if len(self.prev_obs) > self.inference_horizon \
                                        else np.max([self.qs_step - n_step_past,0])
            qs_mean = np.mean(qs_copy, axis=0)
            my_qs = qs_mean[current_qs_idx]

            if verbose:
                print('get_belief_over_states', current_qs_idx, qs_mean)
                print('get_belief_over_states QS:', my_qs)
                    
        return my_qs

    #==== NAVIGATION SETTINGS ====#
    def explo_oriented_navigation(self, inference_algo:str='VANILLA'):
        self.switch_inference_algo(inference_algo)
        self.use_param_info_gain = False
        self.use_states_info_gain = True #Should we
        self.use_utility = False

    def goal_oriented_navigation(self, obs=None, **kwargs):
        inf_algo = kwargs.get('inf_algo', 'MMP')
        self.switch_inference_algo(inf_algo)
        self.update_preference(obs)
        self.use_param_info_gain = False
        self.use_states_info_gain = False #This make it FULLY Goal oriented
        #NOTE: if we want it to prefere this C but still explore a bit once certain about state 
        #(keep exploration/exploitation balanced) keep info gain
        self.use_utility = True
        self.inference_horizon = 4 

    def update_preference(self, obs:list):
        """given a list of observations we fill C with thos as preference. 
        If we have a partial preference over several observations, 
        then the given observation should be an integer < 0, the preference will be a null array 
        """
        if isinstance(obs, list):
            self.update_A_dim_given_obs_3(obs, null_proba=[False]*len(obs))

            C = self._construct_C_prior()

            for modality, ob in enumerate(obs):
                if ob >= 0:
                    self.preferred_ob[modality] = ob
                    ob_processed = utils.process_observation(ob, 1, [self.num_obs[modality]])
                    ob = utils.to_obj_array(ob_processed)
                else:
                    ob = utils.obj_array_zeros([self.num_obs[modality]])
                C[modality] = np.array(ob[0])

            if not isinstance(C, np.ndarray):
                raise TypeError(
                    'C vector must be a numpy array'
                )
            self.C = utils.to_obj_array(C)

            assert len(self.C) == self.num_modalities, f"Check C vector: number of sub-arrays must be equal to number of observation modalities: {agent.num_modalities}"

            for modality, c_m in enumerate(self.C):
                assert c_m.shape[0] == self.num_obs[modality], f"Check C vector: number of rows of C vector for modality {modality} should be equal to {agent.num_obs[modality]}"
        else:
            self.preferred_ob = [-1,-1]
            self.C = self._construct_C_prior()

    def update_agent_state_mapping(self, pose:tuple, ob:list, state_belief:list=None)-> dict:
        """ Dictionnary to keep track of believes and associated obs, usefull for testing purposes mainly"""
        if state_belief is None:
            state = -1
        else:
            state = np.argmax(state_belief)
        #If we already have an ob, let's not squish it with ghost nodes updates
        if pose in self.agent_state_mapping.keys() and self.agent_state_mapping[pose]['ob'] != -1:
            ob[0] = self.agent_state_mapping[pose]['ob']
        self.agent_state_mapping[pose] = {'state' : state , 'ob': ob[0]}
        if len(ob) > 1:
           self.agent_state_mapping[pose]['ob2'] =  ob[1] 
      
        return self.agent_state_mapping
     
    def from_pose_to_idx(self, poses):
        poses_idx = [self.pose_mapping.index(v) if v in self.pose_mapping else ValueError for v in poses]

        if ValueError in poses_idx:
            raise ValueError("unrecognised "+str(poses[poses_idx.index(ValueError)]) +" position in observations")
        
        return poses_idx
    
    def switch_inference_algo(self, algo_type=None):
        if isinstance(algo_type, str):
            self.inference_algo = algo_type

        elif self.inference_algo == "VANILLA":
            self.inference_algo = "MMP" 
        
        else:
            self.inference_algo = "VANILLA" 
        self.inference_params = self.inference_params_dict[self.inference_algo]
        
    #==== NAVIGATION METHODS ====#
   
    def infer_pose(self, action:int, next_possible_actions:list):
        if action in next_possible_actions and self.current_pose !=None:
           self.current_pose = next_p_given_a(self.current_pose, self.possible_actions, action) 
        return self.current_pose

    def infer_action(self, **kwargs):
        observations = kwargs.get('observation', None)
        next_possible_actions = kwargs.get('next_possible_actions', list(self.possible_actions.values()))
        qs_hist = self.qs_hist[-2:]
        #prior = np.pad(qs_hist[-2][0], (0, max(len(qs_hist[-2][0]), len(qs_hist[-1][0])) - len(qs_hist[-2][0])), mode='constant')


        prior = self.get_belief_over_states()
        
        if observations is not None and self.current_pose is None:
        #If self.current_pose is not None then we have step_update that infer state
            
            #NB: Only give obs if state not been inferred before 
            if len(observations) < len(self.A):
                partial_ob = 0
                            
            elif len(observations) == len(self.A):
                partial_ob = None
                if self.current_pose == None:
                    self.current_pose = observations[1]
                observations[1] = self.pose_mapping.index(observations[1])
            
            _, posterior = self.infer_states(observation = observations, distr_obs=False, partial_ob=partial_ob, save_hist=True)
            #print('infer action: self.current_pose', self.current_pose, posterior[0].round(3))
        
        self.use_param_info_gain = False

        #In case we don't have observations.
        posterior = self.get_belief_over_states()
        print('infer action: inferred prior state', posterior[0].round(3))
        q_pi, efe = self.infer_policies(posterior)
        
        action = self.sample_action(next_possible_actions)
        
        #NOTE: What we would expect given prev prior and B transition 
        prior = spm_dot(self.B[0][:, :, int(action)], prior)
        
        return int(action), {
            "qs": posterior[0],
            "qpi": q_pi,
            "efe": efe,
            "bayesian_surprise": utils.bayesian_surprise(posterior[0].copy(), prior),
            }
    
    #==== Update A and B ====#
    
    def update_A_with_data(self,obs:list, state:int)->np.ndarray:
        """Given obs and state, update A entry """
        A = self.A
        
        for dim in range(self.num_modalities ):
            A[dim][:,state] = 0
            A[dim][obs[dim],state] = 1
        self.A = A
        return A

    def update_A_dim_given_obs_3(self, obs:list,null_proba:list=[True]) -> np.ndarray:
        ''' 
        Verify if the observations are new and fit into the current A shape.
        If not, increase A shape in observation (n row) only.
        '''
        A = self.A
        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=A)
        
        # Calculate the maximum dimension increase needed across all modalities
        dim_add = [int(max(0,obs[m] + 1 - num_obs[m])) for m in range(num_modalities)]

        # Update matrices size
        for m in range(num_modalities):
            A[m] = update_A_matrix_size(A[m], add_ob=dim_add[m], null_proba=null_proba[m])
            if self.pA is not None:
                self.pA[m] = utils.dirichlet_like(utils.to_obj_array(A[m]), scale=1)[0]
        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=A)
        self.num_obs = num_obs
        self.A = A
        return A

    def update_B_dim_given_A(self)-> np.ndarray:
        """ knowing A dimension, update B state dimension to match"""
        B = self.B
        add_dim = self.A[0].shape[1]-B[0].shape[1]
        if add_dim > 0: 
            #increase B dim
            B = update_B_matrix_size(B, add= add_dim)
            self.pB = update_B_matrix_size(self.pB, add= add_dim, alter_weights=False)
            if len(self.qs) > 1:
                for seq in self.qs:
                    for subseq in seq:
                        subseq[0] = np.append(subseq[0], [0] * add_dim)
            else:
            
                self.qs[0] = np.append(self.qs[0],[0]*add_dim)
        
        self.num_states = [B[0].shape[0]]
        self.B = B
        return B
    
    def update_A_dim_given_pose(self, pose_idx:int,null_proba:bool=True) -> np.ndarray:
        ''' 
        Verify if the observations are new and fit into the current A shape.
        If not, increase A shape and associate those observations with the newest state generated.
        '''
        A = self.A
        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=A)
        
        # Calculate the maximum dimension increase needed across all modalities
        dim_add = int(max(0,pose_idx + 1 - num_obs[num_modalities-1]))
        # Update matrices size
        #and associate new observations with the newest state generated
        if dim_add > 0:
            A[0] = update_A_matrix_size(A[0], add_state=dim_add, null_proba=null_proba)
            if num_modalities > 1:
                A[1] = update_A_matrix_size(A[1], add_ob=dim_add, add_state=dim_add, null_proba=null_proba)
                #we search the first fully null or normed column (thus no link between state -> ob) #THIS IS MAINLY FOR SECURITY
                columns_wthout_data = np.sort(np.append(np.where(np.all(A[1] == 1/A[1].shape[0], axis=0))[0], np.where(np.all(A[1] == 0, axis=0))[0]))
                A[1][:, columns_wthout_data[0]] = 0
                A[1][pose_idx, columns_wthout_data[0]] = 1
                self.num_obs[1] = A[1].shape[0]

            if self.pA is not None:
                self.pA = utils.dirichlet_like(utils.to_obj_array(A), scale=1)
                    
        self.num_states = [A[0].shape[1]]
        self.A = A
        return A
    
    def update_C_dim(self):
        if self.C is not None:
            num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A=self.A) 
            for m in range(num_modalities):
                if self.C[m].shape[0] < num_obs[m]:
                    self.C[m] = np.append(self.C[m], [0]*(num_obs[m]- self.C[m].shape[0]))
                    
    #==== update Believes ====#
    def update_A_belief(self,obs:list)->None:
        #UPDATE A given all observations
        #IDENTIFY WHERE THE AGENT BELIEVES TO BE
        _, Qs = self.infer_states(obs, distr_obs=False) 
        self.update_A(obs, Qs)
        self.update_A(obs, Qs) #twice to increase effect (not mandatory)

    def update_believes_v2(self, Qs:list, action:int, obs:list)-> None:
        #UPDATE B
        if len(self.qs_hist) > 0:#secutity check
            qs_hist = self.get_belief_over_states(self.qs_hist[-1])
            qs_hist[0] = np.append(qs_hist[0],[0]*\
                                   (len(Qs[0])-len(qs_hist[0])))
            self.update_B(Qs, qs_hist, action, lr_pB = 10) 
            #2 WAYS TRANSITION UPDATE (only if T to diff state)
            if np.argmax(qs_hist[0]) != np.argmax(Qs[0]):
                a_inv = reverse_action(self.possible_actions, action)
                self.update_B(qs_hist, Qs, a_inv, lr_pB = 7)
        self.update_A_belief(obs)
    
    def add_ghost_node_v3(self, qs:np.ndarray, p_idx:int, possible_next_actions:list)-> None:
        ''' 
        For each new pose observation, add a ghost state and update the estimated transition and observation for that ghost state.
        '''
        print('Ghost nodes process:')
        pose = self.pose_mapping[p_idx]

        for action in self.possible_actions.values():
            if action not in possible_next_actions: #this mean this action is not deemed possible
                self.update_B(qs, qs, action, lr_pB = 10)
            else:
                n_pose = next_p_given_a(pose, self.possible_actions, action)
                if n_pose not in self.pose_mapping:
                    self.pose_mapping.append(n_pose)
                    p_idx = self.pose_mapping.index(n_pose)
                    print('a',action,'n pose', n_pose, p_idx)
                    self.update_A_dim_given_pose(p_idx, null_proba=False) #we only update pose ob and assign a state to this ob
                    self.update_B_dim_given_A()
                    _, hypo_qs = self.infer_states([p_idx], np.array([action]), partial_ob=1, save_hist=False)
                    if len(qs[0]) < len(hypo_qs[0]):
                        qs[0] = np.append(qs[0],[0]*(len(hypo_qs[0])-len(qs[0])))
                    self.update_B(hypo_qs, qs, action, lr_pB = 5) 
                    self.update_agent_state_mapping(n_pose, [-1], hypo_qs[0])
                    a_inv = reverse_action(self.possible_actions, action)
                    self.update_B(qs, hypo_qs, a_inv, lr_pB = 3)
        
    
    #==== PYMDP modified methods ====#
    def reset(self, init_qs:np.ndarray=None, start_pose:tuple=None):
        """
        Resets the posterior beliefs about hidden states of the agent to a uniform distribution, and resets time to first timestep of the simulation's temporal horizon.
        Returns the posterior beliefs about hidden states.

        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
           Initialized posterior over hidden states. Depending on the inference algorithm chosen and other parameters (such as the parameters stored within ``edge_handling_paramss),
           the resulting ``qs`` variable will have additional sub-structure to reflect whether beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `, the indexing structure of ``qs`` is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``. In this case, the returned ``qs`` will only have entries filled out for the first timestep, i.e. for ``q[p_idx][0]``, for all 
            policy-indices ``p_idx``. Subsequent entries ``q[:][1, 2, ...]`` will be initialized to empty ``numpy.ndarray`` objects.
        """

        self.curr_timestep = 0
        self.action = None
        self.prev_actions = None
        self.prev_obs = []
        self.qs_step = 0
     
        self.current_pose = start_pose
        if init_qs is None:
            
            self.D = self._construct_D_prior()
           
            if hasattr(self, "q_pi_hist"):
                self.q_pi_hist = []

            if hasattr(self, "qs_hist"):
                self.qs_hist = []
            
            if self.inference_algo == 'VANILLA':
                self.qs = utils.obj_array_uniform(self.num_states)
            else: # in the case you're doing MMP (i.e. you have an inference_horizon > 1), we have to account for policy- and timestep-conditioned posterior beliefs
                self.qs = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                
                    self.qs[p_i] = utils.obj_array_uniform(\
                        [self.num_states] * (self.inference_horizon + self.policy_len + 1)) # + 1 to include belief about current timestep
                    #self.qs[p_i][0] = utils.obj_array_uniform(self.num_states)
                
                first_belief = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                    first_belief[p_i] = copy.deepcopy(self.D) 
                
                if self.edge_handling_params['policy_sep_prior']:
                    self.set_latest_beliefs(last_belief = first_belief)
                else:
                    self.set_latest_beliefs(last_belief = self.D)
        
        else:
            self.qs = init_qs

        return self.qs
    
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
            Posterior Dirichlet parameters over transition self (same shape as ``B``), after having updated it with state beliefs and actions.
        """
        
        if lr_pB is None:
            lr_pB = self.lr_pB

        qB = update_state_likelihood_dirichlet(
            self.pB,
            self.B,
            [action],
            qs,
            qs_prev,
            lr_pB,
            self.factors_to_learn
        )

        self.pB = qB # set new prior to posterior
        self.B = utils.norm_dist_obj_arr(qB)  # take expected value of posterior Dirichlet parameters to calculate posterior over B array
        return qB

    def update_A(self, obs, qs=None):
        """
        Update approximate posterior beliefs about Dirichlet parameters that parameterise the observation likelihood or ``A`` array.

        Parameters
        ----------
        observation: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores the index of the discrete
            observation for modality ``m``.

        Returns
        -----------
        qA: ``numpy.ndarray`` of dtype object
            Posterior Dirichlet parameters over observation self (same shape as ``A``), after having updated it with observations.
        """
        if qs is None:
            qs = self.qs
        qA = update_obs_likelihood_dirichlet(
            self.pA, 
            self.A, 
            obs, 
            qs, 
            self.lr_pA, 
            self.modalities_to_learn
        )

        self.pA = qA # set new prior to posterior
        self.A = utils.norm_dist_obj_arr(qA) # take expected value of posterior Dirichlet parameters to calculate posterior over A array

        return qA
    
    def infer_states(self, observation, action= None ,save_hist=True, distr_obs = False, partial_ob=None):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

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
        # print('infer state',self.inference_algo, action)
        observation = tuple(observation) if not distr_obs else observation
        if save_hist:
            self.prev_obs.append(observation)
            observations_hist = self.prev_obs
        else:
            observations_hist = self.prev_obs.copy()
            observations_hist.append(observation)

        if action != None:
            if self.prev_actions != None:
                prev_actions = self.prev_actions.copy()
            else:
                prev_actions = []
            prev_actions.append(action)
        else:
            prev_actions = self.prev_actions

        if len(observations_hist) > self.inference_horizon:
            latest_obs = observations_hist[-self.inference_horizon:]
            latest_actions = prev_actions[-(self.inference_horizon-1):]
        else:
            latest_obs = observations_hist
            latest_actions = prev_actions
        
        if partial_ob is None and len(latest_obs[0]) != len(latest_obs[-1]):
            self.qs_step = 0
            self.prev_actions = None
            self.prev_obs = []
            if save_hist:
                self.prev_obs = [latest_obs[-1]]
        
            
            latest_obs = [latest_obs[-1]]
            latest_actions = self.prev_actions
        if self.inference_algo == "VANILLA":
            if self.action is not None:
                qs = self.get_belief_over_states() #we don't want to consider current obs to selest qs
                empirical_prior = control.get_expected_states(
                    qs, self.B, self.action.reshape(1, -1) #type: ignore
                )[0]
            else:
                self.D = self._construct_D_prior() #self.D
                empirical_prior = self.D

            qs = update_posterior_states(
            self.A,
            observation,
            empirical_prior,
            partial_ob,
            **self.inference_params
            )
            F = 0
            mean_qs_over_policies = qs.copy()
            qs_step = 0
        elif self.inference_algo == "MMP":

            if not hasattr(self, "qs"):
                self.reset()
    
            prior = self.latest_belief

            #MMP 
            if isinstance(prior[0][0], np.ndarray):  # Check if nested array
                for i in range(len(prior)):
                    if len(prior[i][0]) < self.num_states[0]:
                        prior[i][0] = np.append(prior[i][0], [0] * (self.num_states[0] - len(prior[i][0])))
                self.latest_belief = prior
                if not self.edge_handling_params['policy_sep_prior']:
                    prior = np.mean(prior, axis=0)
      
            
            elif len(prior[0]) < self.num_states[0]:
                prior[0] = np.append(prior[0], [0] * (self.num_states[0] - len(prior[0])))
                self.latest_belief = prior
                self.D = self._construct_D_prior()
     

            # print('latest_obs',latest_obs)
            # print('latest_actions',latest_actions)
            # print('partial_ob', partial_ob)
            # print('prior', self.latest_belief)
            qs, F = update_posterior_states_full(
                self.A,
                self.B,
                latest_obs,
                self.policies, 
                latest_actions, 
                prior = prior, 
                policy_sep_prior = self.edge_handling_params['policy_sep_prior'],
                partial_ob = partial_ob,
                **self.inference_params
            )
  
            selected_qs = [q[:self.policy_len + 1] for q in qs]
            mean_qs = np.mean(selected_qs, axis=0)
            qs_step = len(latest_obs)-1
            mean_qs_over_policies = mean_qs[qs_step]
            # print('current full qs mean', mean_qs)
            # print('current_qs',mean_qs_over_policies, 'qs idx', self.qs_step, 'save hist', save_hist)
        if save_hist:
            self.F = F # variational free energy of each policy  
            self.qs_step = qs_step
            if hasattr(self, "qs_hist"):
                self.qs_hist.append(qs)
            self.qs = qs

        return qs, mean_qs_over_policies

    def infer_policies(self, qs=None):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """
        if qs is None:
            qs = self.qs

        if self.inference_algo == "VANILLA":
            
            q_pi, G = update_posterior_policies(
                qs,
                self.A,
                self.B,
                self.C,
                self.policies,
                self.use_utility,
                self.use_states_info_gain,
                self.use_param_info_gain,
                self.pA,
                self.pB,
                E = self.E,
                gamma = self.gamma,
                diff_policy_len = self.lookahead_distance
            )
        elif self.inference_algo == "MMP":
            # if qs is None:
            #     future_qs_seq = self.get_future_qs()

            q_pi, G = update_posterior_policies_full(
                qs, #future_qs_seq
                self.A,
                self.B,
                self.C,
                self.policies,
                self.use_utility,
                self.use_states_info_gain,
                self.use_param_info_gain,
                self.latest_belief,
                self.pA,
                self.pB,
                F = self.F,
                E = self.E,
                gamma = self.gamma,
                diff_policy_len = self.lookahead_distance
            )

        if hasattr(self, "q_pi_hist"):
            self.q_pi_hist.append(q_pi)
            if len(self.q_pi_hist) > self.inference_horizon:
                self.q_pi_hist = self.q_pi_hist[-(self.inference_horizon-1):]
            

        self.q_pi = q_pi
        self.G = G
        return q_pi, G
    
    def sample_action(self, possible_first_actions:list=None):
        """
        Sample or select a discrete action from the posterior over control states.
        This function both sets or cachés the action as an internal variable with the agent and returns it.
        This function also updates time variable (and thus manages consequences of updating the moving reference frame of beliefs)
        using ``self.step_time()``.

        
        Returns
        ----------
        action: 1D ``numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        """
        if possible_first_actions != None:
            #Removing all policies leading us to uninteresting action.
            policies, q_pi = zip(*[(policy, self.q_pi[p_id]) for p_id, policy \
                                   in enumerate(self.policies) if policy[0] in possible_first_actions])
        else:
            policies =  self.policies
            q_pi = self.q_pi

        if self.sampling_mode == "marginal":
            action = control.sample_action(
                q_pi, policies, self.num_controls, action_selection = self.action_selection, alpha = self.alpha
            )
        elif self.sampling_mode == "full":
            action = control.sample_policy(q_pi, policies, self.num_controls,
                                           action_selection=self.action_selection, alpha=self.alpha)

        self.action = action

        self.step_time()

        return action
    
    #==== BELIEVES PROCESS UPDATE ====#
    def agent_step_update(self, action, observations = [0,(0,0)], possible_next_actions:list=[0,1,2,3]):
        
        #We only update A and B if we have inferred a current pose
        #Thus until doubt over current loc is not solved, we don't update internal self
        if self.current_pose != None:
            print('Updating model given observations', observations)
            ob = observations[0]
            if len(observations) > 1:
                self.current_pose = observations[1]
        
            pose = self.current_pose

            if pose not in self.pose_mapping:
                self.pose_mapping.append(pose)
            p_idx = self.pose_mapping.index(pose)
            # prev_state_size = agent.num_states[0]
            #3. UPDATE A AND B DIM WITH THOSE DATA
            self.update_A_dim_given_obs_3([ob,p_idx], null_proba=[False,False])
            self.update_B_dim_given_A()
            # new_state_size = agent.num_states[0]
            #4. UPDATE BELIEVES GIVEN OBS
            _, Qs = self.infer_states([ob,p_idx], distr_obs=False, save_hist=False)
            print('prior on believed state; action', action, 'colour_ob:', ob, 'inf pose:',pose,'belief:', Qs[0].round(3))
            
            #4.5 UPDATE A AND B WITH THOSE BELIEVES
            self.update_believes_v2(Qs, action, [ob,p_idx])
            qs = self.get_belief_over_states()
            # print('Defined belief after A and B update:', qs[0].round(3))
            self.update_agent_state_mapping(pose, [ob,p_idx], qs[0])
            #ADD KNOWLEDGE WALL T OR GHOST NODES
            #inv_action = reverse_action(self.possible_actions, action) #just to gain some computation time
            self.add_ghost_node_v3(qs,p_idx, possible_next_actions)

            #This is not mandatory, just a gain of time
            if 'STAY' in self.possible_actions:
                self.B[0] = set_stationary(self.B[0], self.possible_actions['STAY'])
            self.update_C_dim()








