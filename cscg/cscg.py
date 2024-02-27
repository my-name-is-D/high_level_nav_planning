from __future__ import print_function
from builtins import range
import numpy as np
import numba as nb
from tqdm import trange
import sys
import copy
from ours.pymdp.agent import Agent
from ours.pymdp import utils
from ours.pymdp.maths import spm_dot
from envs.modules import next_p_given_a

import pandas as pd

def validate_seq(x, a, n_clones=None):
    """Validate an input sequence of observations x and actions a"""
    assert len(x) == len(a) > 0
    assert len(x.shape) == len(a.shape) == 1, "Flatten your array first"
    assert x.dtype == a.dtype == np.int64
    assert 0 <= x.min(), "Number of emissions inconsistent with training sequence"
    if n_clones is not None:
        assert len(n_clones.shape) == 1, "Flatten your array first"
        assert n_clones.dtype == np.int64
        assert all(
            [c > 0 for c in n_clones]
        ), "You can't provide zero clones for any emission"
        n_emissions = n_clones.shape[0]
        assert (
            x.max() < n_emissions
        ), "Number of emissions inconsistent with training sequence"


def datagen_structured_obs_room(
    room,
    start_r=None,
    start_c=None,
    no_left=[],
    no_right=[],
    no_up=[],
    no_down=[],
    length=10000,
    seed=42,
):
    """room is a 2d numpy array. inaccessible locations are marked by -1.
    start_r, start_c: starting locations

    In addition, there are invisible obstructions in the room
    which disallows certain actions from certain states.

    no_left:
    no_right:
    no_up:
    no_down:

    Each of the above are list of states from which the corresponding action is not allowed.

    """
    np.random.seed(seed)
    H, W = room.shape
    if start_r is None or start_c is None:
        start_r, start_c = np.random.randint(H), np.random.randint(W)

    actions = np.zeros(length, int)
    x = np.zeros(length, int)  # observations
    rc = np.zeros((length, 2), int)  # actual r&c

    r, c = start_r, start_c
    x[0] = room[r, c]
    rc[0] = r, c

    count = 0
    while count < length - 1:

        act_list = [0, 1, 2, 3]  # 0: left, 1: right, 2: up, 3: down
        if (r, c) in no_left:
            act_list.remove(0)
        if (r, c) in no_right:
            act_list.remove(1)
        if (r, c) in no_up:
            act_list.remove(2)
        if (r, c) in no_down:
            act_list.remove(3)

        a = np.random.choice(act_list)

        # Check for actions taking out of the matrix boundary.
        prev_r = r
        prev_c = c
        if a == 0 and 0 < c:
            c -= 1
        elif a == 1 and c < W - 1:
            c += 1
        elif a == 2 and 0 < r:
            r -= 1
        elif a == 3 and r < H - 1:
            r += 1

        # Check whether action is taking to inaccessible states.
        temp_x = room[r, c]
        if temp_x == -1:
            r = prev_r
            c = prev_c
            pass

        actions[count] = a
        x[count + 1] = room[r, c]
        rc[count + 1] = r, c
        count += 1

    return actions, x, rc


class CHMM(object):
    def __init__(self, n_clones, x, a, possible_actions, pseudocount=0.0, \
                 dtype=np.float32, seed=42, set_stationary_B=True, ob_ambiguity=True):
        """Construct a CHMM objct. n_clones is an array where n_clones[i] is the
        number of clones assigned to observation i. x and a are the observation sequences
        and action sequences, respectively."""
        np.random.seed(seed)
        self.n_clones = n_clones
        validate_seq(x, a, self.n_clones)
        assert pseudocount >= 0.0, "The pseudocount should be positive"
        #print("Average number of clones:", n_clones.mean())
        self.pseudocount = pseudocount
        self.dtype = dtype
        n_states = self.n_clones.sum()
        self.n_actions = a.max() + 1
        self.possible_actions = possible_actions
        self.C = np.random.rand(self.n_actions, n_states, n_states).astype(dtype)
        self.Pi_x = np.ones(n_states) / n_states
        self.Pi_a = np.ones(self.n_actions) / self.n_actions
        self.update_T()
        self.agent_state_mapping = {}
        self.pose_mapping = []
        self.states = []
        self.set_stationary_B = set_stationary_B
        A, B = self.extract_AB(reduce=False, do_add_stationary= False, do_add_death=False)
        
        self.agent = Agent(
            A=A, B=B, action_selection="stochastic", policy_len=1
        )
        self.prev_action = None
        self.preferred_states = []
        self.preferred_ob = [-1]
        self.ob_ambiguity = ob_ambiguity
        self.current_pose = (0,0)

    def get_belief_over_states(self):
        return self.agent.qs
    
    def infer_pose(self, action, next_possible_actions):
        if action in next_possible_actions:
           self.current_pose = next_p_given_a(self.current_pose, self.possible_actions, action) 
        return self.current_pose
    
    def update_agent_state_mapping(self, pose:tuple, ob:int, state_belief:list=None)-> dict:
        """ Dictionnary to keep track of believes and associated obs, usefull for testing purposes mainly"""
        if state_belief is None:
            state = -1
        else:
            state = np.argmax(state_belief)
        
        # If pose already exists in the mapping, update the information
        if pose in self.agent_state_mapping.keys():
            existing_info = self.agent_state_mapping[pose]
            #if we have an ob, let's keep it
            # if existing_info['ob'] != -1:
            #     ob = existing_info['ob']
            state_ob = existing_info.get('state_ob', -1)
            state = existing_info.get('state', -1)
        else:
            # Add new pose to mapping
            self.pose_mapping.append(pose)
            state_ob = len(self.agent_state_mapping) 
        p_idx = self.pose_mapping.index(pose)
            
        # Update agent state mapping
        self.agent_state_mapping[pose] = {'state': state, 'state_ob': state_ob, 'ob': ob, 'pose_idx': p_idx}
        if self.ob_ambiguity:
            state_ob = ob

        return state_ob
    
    def infer_action(self, **kwargs):
        obs=kwargs.get('observation',[-1])
        next_possible_actions = kwargs.get('next_possible_actions',[*range(self.n_actions)])
        random = kwargs.get('random_policy', False)

        if len(obs) > 1:
            state_ob = self.update_agent_state_mapping(obs[1], obs[0])
            print('state_ob', state_ob, 'for obs', obs)
            obs = [state_ob] 
            #Pose is an unambigious ob, since we give an int to cscg, still
            #for future changes in method. let's keep separated.
        elif isinstance(obs[0], tuple):
            state_ob = self.update_agent_state_mapping(obs[0], obs[0])
            obs = [state_ob]  

        if random:
            return np.random.choice(next_possible_actions), {
                "qs": self.agent.qs[0],
                "bayesian_surprise": 0,
                }
        
        prior = self.agent.qs[0].copy()
        if self.prev_action is not None:
            prior = spm_dot(self.agent.B[0][:, :, self.prev_action], prior)
            #TODO: CHECK PRIOR
        
        qs = self.agent.infer_states(obs)

        action = int(self.get_action(qs,next_possible_actions))

        self.prev_action = action

        self.agent.action = np.array([self.prev_action])
        self.agent.step_time()

        posterior = self.agent.qs[0].copy()
        return action, {
            "qs": self.agent.qs[0],
            "bayesian_surprise": utils.bayesian_surprise(posterior, prior),
            }
    
    def get_pose_mapping(self):
        return self.pose_mapping
    
    def get_action(self, qs,next_possible_actions):
        # states = np.where(qs[0][:-1] > 1e-4)[0]
        states = np.where(qs[0] > 1e-4)[0]
        pq = qs[0][states]

        plans = []
        n_states = []
        for i, s in enumerate(states):
            # pi_x = np.zeros_like(qs[0][:-1])
            pi_x = np.zeros_like(qs[0])
            pi_x[s] = 1
            #if s not in chmm.preferred_states:
            if s in self.states:
                actions, states = self.observation_bridge(pi_x, max_steps=15) #consider preference
            
                if actions[0] == -1:
                    pq[i] = 0
                elif states[0] in self.states:
                    pq[i] = pq[i] * 1.5
            else:
                actions = [-1]  # just go forward if its already in preferred
                states = [-1]
                pq[i] = 0
            
            plans.append(actions[0])
            n_states.append(states[0])

        norm = pq.sum() 
        if norm <= 0:
            pq = [1/len(pq)]*len(pq)
            action = np.random.choice(next_possible_actions)
        else:
            pq /= norm
            action = np.random.choice(plans, p=pq)
        return action
    
    def from_pose_to_idx(self, poses):
        poses_idx = [self.pose_mapping.index(v) if v in self.pose_mapping else ValueError for v in poses]

        if ValueError in poses_idx:
            raise ValueError("unrecognised "+str(poses[poses_idx.index(ValueError)]) +" position in observations")
        return poses_idx
    
    def from_obs_to_ob(self, obs):
        """obs : [colour, pose] or [pose] 
        Transform them into state_obs by mapping to colour+pose obs. 
        Or pose index if [pose]
        That is if no ambiguous observation.
        else did not implement
        """
        if not self.ob_ambiguity:
            if obs.shape[1] == 2 :
                observations = [] 
                # Create a dictionary for quick lookup based on 'ob' and 'pose'
                state_mapping_lookup = {(info['ob'], pose): info \
                                        for pose, info in self.agent_state_mapping.items()}
                for i, (ob, pose) in enumerate(obs):
                    # Check if there is a matching entry in the state_mapping_lookup
                    if (ob, pose) in state_mapping_lookup:
                        observations.append(state_mapping_lookup[(ob, pose)]['state_ob'])
                    else:
                        raise ValueError('Observations ' + str((ob, pose))+ 'not existing in state_mapping')
                
            elif isinstance(obs[0], tuple):
                poses_idx = self.from_pose_to_idx(obs)
                observations = poses_idx
        return observations

    def get_agent_state_mapping(self, x=None,a=None, agent_pose=None):
        if x is not None:
            self.define_agent_state_mapping(x,a, agent_pose)
        return self.agent_state_mapping
    
    def set_agent_state_mapping(self, x, agent_pose=None):
        for id, ob in enumerate(x):
            if isinstance(ob, np.ndarray):
                self.update_agent_state_mapping(ob[1], ob[0])
            else:
                self.update_agent_state_mapping(agent_pose[id], ob[0])
        return self.agent_state_mapping
    

    def get_B(self, set_stationary_B=None):
        B = copy.deepcopy(self.T)
        rearranged_B = np.transpose(B,(2,1,0))
        if set_stationary_B == True or (set_stationary_B is None and self.set_stationary_B == True):
            rearranged_B = set_stationary(rearranged_B)
        
        return rearranged_B
    
    def get_A(self,reduce=True, do_add_death=False):
        death = int(do_add_death)
        n_obs = len(self.n_clones)

        T = self.get_B()
        unreduced_n_states = T.shape[0]
        # for a in range(T.shape[2]):
        #     print('T',a,pd.DataFrame(T[:,:,a], index=list(range(0,T.shape[0])), columns=list(range(0,T.shape[1])), dtype=float))
        if reduce:
            v = T.sum(axis=2).sum(axis=1).nonzero()[0]

        # A matrix = likelihood matrix, unreduced matrix and uniform probabilities
        state_loc = np.hstack(
            (np.array([0], dtype=self.n_clones.dtype), self.n_clones)
        ).cumsum()

        A = np.zeros((n_obs, unreduced_n_states + death))
        for i in range(n_obs - int(death)):
            s, f = state_loc[i : i + 2]
            A[i, s:f] = 1.0
      
        # Direct mapping of state to a death observation
        if do_add_death:
            A[-1, -1] = 1.0
      
        if reduce and do_add_death:
            v = np.concatenate([v, np.array([-1])])

        # Only consider the reduced states
        if reduce:
            A = A[:, v]

        # Normalize over state: Sum_s P(o|s) = 1
        A /= A.sum(axis=0, keepdims=True)
        return [A]

    def extract_AB(self, reduce=False, do_add_stationary=False, do_add_death=True):
        """ death --> walls?"""
        death = int(do_add_death)
        
        T = self.get_B()
        
        if reduce:
            v = T.sum(axis=2).sum(axis=1).nonzero()[0]
            T = T[v, :][:, v]

        # Transition matrix
        B = np.zeros((T.shape[0] + death, T.shape[0] + death, T.shape[2]))
        B[: T.shape[0], : T.shape[1]] = T
        if do_add_death:
            B = add_death(B)
        if do_add_stationary:
            B = add_stationary(B)
        # Normalize this dude
        B /= B.sum(axis=0, keepdims=True)

        A = self.get_A(reduce, do_add_death)[0]
       
        return A, B

    def get_n_states(self):
        
        return len(self.states)

    def define_agent_state_mapping(self, x, a, agent_poses):
        
        a = np.array(a).flatten().astype(int)
        obs = self.format_observations(x)
        states = self.decode(obs, a)[1]
         
        values = []
        for p_idx in range(len(agent_poses)-1, 0, -1):
            pose = tuple(agent_poses[p_idx])
            state = states[p_idx] 
            if state not in values or pose not in self.agent_state_mapping.keys() :
                if pose not in self.agent_state_mapping.keys() :
                    if pose not in self.pose_mapping:
                        self.pose_mapping.append(pose)
                    state_ob = len(self.agent_state_mapping)
                    self.agent_state_mapping[pose] = {'state_ob':state_ob}
               
                self.agent_state_mapping[pose]['state'] = state
                if isinstance(x[0], np.ndarray):
                    self.agent_state_mapping[pose]['ob'] = x[p_idx,0]
                    self.agent_state_mapping[pose]['pose_idx'] = x[p_idx,1]
                else:
                    self.agent_state_mapping[pose]['ob'] = obs[p_idx]
                values.append(state)
        self.agent_state_mapping = dict(sorted(self.agent_state_mapping.items(), key=lambda x: x[1]['state']))
        return 
   

    def agent_step_update(self,action,obs,next_possible_actions):
        """ Do nothing, wait for full motion?"""
        return
    
    def goal_oriented_navigation(self, obs, **kwargs):
        if -1 in obs:
            obs.remove(-1)

        test_obs = obs

        if not self.ob_ambiguity:
            observation = []
            for pose, info in self.agent_state_mapping.items():
                if len(obs) == 2 :
                    if info['ob'] == obs[0] and pose == obs[1]:
                        observation = [self.agent_state_mapping[pose]['state_ob']]

                elif info['ob'] == obs[0]:
                    observation.append(self.agent_state_mapping[pose]['state_ob'])
        
            obs = observation
        print('received obs', test_obs, 'modified obs', obs)
        self.update_preference(obs) 
    
    def explo_oriented_navigation(self, **kwargs):
        self.update_preference()

    def update_preference(self, obs:list=None):
        """given a list of observations we fill C with thos as preference. 
        If we have a partial preference over several observations, 
        then the given observation should be an integer < 0, the preference will be a null array 
        """
        if isinstance(obs, list):
            C = self.agent._construct_C_prior()
            A = self.get_A(reduce=False)[0]
            
            preferred_ob = []
            #Only 1 modality, but several goal obs possible
            for id, ob in enumerate(obs):
                if ob >= 0:
                    if len(self.preferred_ob) == 1 and self.preferred_ob[0] == -1:
                        self.preferred_ob = []
                    self.preferred_ob.append(ob)
                    self.preferred_states += list(A[ob, :].nonzero()[0])
                    ob_processed = utils.process_observation(ob, 1, [self.agent.num_obs[0]])
                    ob = utils.to_obj_array(ob_processed)
                    #saving several goals
                    if id ==0:
                        ob_modality = ob
                    else:
                        ob_modality+= ob
                else:
                    ob_modality = utils.obj_array_zeros([self.agent.num_obs[0]])
                preferred_ob.append(ob_modality[0])
                
            C = np.array(preferred_ob, dtype=object)

            if not isinstance(C, np.ndarray):
                raise TypeError(
                    'C vector must be a numpy array'
                )
            self.agent.C = utils.to_obj_array(C)

            # assert len(self.agent.C) == self.agent.num_modalities, f"Check C vector: number of sub-arrays must be equal to number of observation modalities: {agent.num_modalities}"

            # for modality, c_m in enumerate(self.agent.C):
            #     assert c_m.shape[0] == self.agent.num_obs[modality], f"Check C vector: number of rows of C vector for modality {modality} should be equal to {agent.num_obs[modality]}"
            
        else:
            self.agent.C = self.agent._construct_C_prior()
            self.preferred_states = []
            self.preferred_ob = [-1]

    def observation_bridge(self, belief_over_states, max_steps=100):
        
        ret = forward_mp_all_multiple_states(
            self.T.transpose(0, 2, 1),
            belief_over_states,
            self.Pi_a,
            self.n_clones,
            self.preferred_states,
            max_steps,
        )
        #print('belief_over_states',np.argmax(belief_over_states), 'mess_fwd, pref state' ,ret[1:] )
        if ret:
            log2_lik, mess_fwd, selected_state = ret
            s_a = backtrace_all(
                self.T, self.Pi_a, self.n_clones, mess_fwd, selected_state
            )
        else:
            return [-1], [-1]
        return s_a

    def format_observations(self,x):
        if isinstance(x[0], np.ndarray):
            if isinstance(x[0][1], tuple):
                x[:,1] = self.from_pose_to_idx(x[:,1])
            if np.max(x[:,1]) > len(self.n_clones):
                raise 'Observation value above agent n_clones set capacity'
            x = np.array(x[:,1]).flatten().astype(np.int64)
        else:
            x = np.array(x).flatten().astype(np.int64)
        return x
    
    def update_T(self):
        """Update the transition matrix given the accumulated counts matrix."""
        self.T = self.C + self.pseudocount
        norm = self.T.sum(2, keepdims=True)
        norm[norm == 0] = 1
        self.T /= norm

        
    # def update_T(self):
    #     self.T = self.C + self.pseudocount
    #     norm = self.T.sum(2, keepdims=True)  # old model (conditional on actions)
    #     norm[norm == 0] = 1
    #     self.T /= norm
    #     norm = self.T.sum((0, 2), keepdims=True)  # new model (generates actions too)
    #     norm[norm == 0] = 1
    #     self.T /= norm

    def update_E(self, CE):
        """Update the emission matrix."""
        E = CE + self.pseudocount
        norm = E.sum(1, keepdims=True)
        norm[norm == 0] = 1
        E /= norm
        return E

    def bps(self, x, a):
        """Compute the log likelihood (log base 2) of a sequence of observations and actions."""
        validate_seq(x, a, self.n_clones)
        log2_lik = forward(self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a)[0]
        return -log2_lik

    def bpsE(self, E, x, a):
        """Compute the log likelihood using an alternate emission matrix."""
        validate_seq(x, a, self.n_clones)
        log2_lik = forwardE(
            self.T.transpose(0, 2, 1), E, self.Pi_x, self.n_clones, x, a
        )
        return -log2_lik

    def bpsV(self, x, a):
        validate_seq(x, a, self.n_clones)
        log2_lik = forward_mp(
            self.T.transpose(0, 2, 1), self.Pi_x, self.n_clones, x, a
        )[0]
        return -log2_lik

    def decode(self, x, a):
        """Compute the MAP assignment of latent variables using max-product message passing."""
        x = self.format_observations(x)
        log2_lik, mess_fwd = forward_mp(
            self.T.transpose(0, 2, 1),
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
        return -log2_lik, states

    def decodeE(self, E, x, a):
        """Compute the MAP assignment of latent variables using max-product message passing
        with an alternative emission matrix."""
        log2_lik, mess_fwd = forwardE_mp(
            self.T.transpose(0, 2, 1),
            E,
            self.Pi_x,
            self.n_clones,
            x,
            a,
            store_messages=True,
        )
        states = backtraceE(self.T, E, self.n_clones, x, a, mess_fwd)
        return -log2_lik, states

    def learn_em_T(self, x, a, n_iter=100, term_early=True):
        """Run EM traself.update_T()ining, keeping E deterministic and fixed, learning T"""
        x = self.format_observations(x)
        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for it in pbar:
            # E
            log2_lik, mess_fwd = forward(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backward(self.T, self.n_clones, x, a)
            updateC(self.C, self.T, self.n_clones, mess_fwd, mess_bwd, x, a)
            # M
            self.update_T()
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                if term_early:
                    break
            log2_lik_old = log2_lik.mean()
        return convergence

    def learn_viterbi_T(self, x, a, n_iter=100):
        """Run Viterbi training, keeping E deterministic and fixed, learning T"""
        x = self.format_observations(x)

        sys.stdout.flush()
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for it in pbar:
            # E
            log2_lik, mess_fwd = forward_mp(
                self.T.transpose(0, 2, 1),
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            states = backtrace(self.T, self.n_clones, x, a, mess_fwd)
            self.C[:] = 0
            for t in range(1, len(x)):
                aij, i, j = (
                    a[t - 1],
                    states[t - 1],
                    states[t],
                )  # at time t-1 -> t we go from observation i to observation j
                self.C[aij, i, j] += 1.0
            # M
            self.update_T()

            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        
        states = self.decode(x, a)[1]
        self.states = np.unique(states)
        if self.set_stationary_B:
            self.T = set_T_stationary(self.T)
        return convergence

    def learn_em_E(self, x, a, n_iter=100, pseudocount_extra=1e-20):
        """Run Viterbi training, keeping T fixed, learning E"""
        sys.stdout.flush()
        n_emissions, n_states = len(self.n_clones), self.n_clones.sum()
        CE = np.ones((n_states, n_emissions), self.dtype)
        E = self.update_E(CE + pseudocount_extra)
        convergence = []
        pbar = trange(n_iter, position=0)
        log2_lik_old = -np.inf
        for it in pbar:
            # E
            log2_lik, mess_fwd = forwardE(
                self.T.transpose(0, 2, 1),
                E,
                self.Pi_x,
                self.n_clones,
                x,
                a,
                store_messages=True,
            )
            mess_bwd = backwardE(self.T, E, self.n_clones, x, a)
            updateCE(CE, E, self.n_clones, mess_fwd, mess_bwd, x, a)
            # M
            E = self.update_E(CE + pseudocount_extra)
            convergence.append(-log2_lik.mean())
            pbar.set_postfix(train_bps=convergence[-1])
            if log2_lik.mean() <= log2_lik_old:
                break
            log2_lik_old = log2_lik.mean()
        return convergence, E

    def sample(self, length):
        """Sample from the CHMM."""
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)
        sample_x = np.zeros(length, dtype=np.int64)
        sample_a = np.random.choice(len(self.Pi_a), size=length, p=self.Pi_a)

        # Sample
        p_h = self.Pi_x
        for t in range(length):
            h = np.random.choice(len(p_h), p=p_h)
            sample_x[t] = np.digitize(h, state_loc) - 1
            p_h = self.T[sample_a[t], h]
        return sample_x, sample_a

    def sample_sym(self, sym, length):
        """Sample from the CHMM conditioning on an inital observation."""
        # Prepare structures
        assert length > 0
        state_loc = np.hstack(([0], self.n_clones)).cumsum(0)

        seq = [sym]

        alpha = np.ones(self.n_clones[sym])
        alpha /= alpha.sum()

        for _ in range(length):
            obs_tm1 = seq[-1]
            T_weighted = self.T.sum(0)

            long_alpha = np.dot(
                alpha, T_weighted[state_loc[obs_tm1] : state_loc[obs_tm1 + 1], :]
            )
            long_alpha /= long_alpha.sum()
            idx = np.random.choice(np.arange(self.n_clones.sum()), p=long_alpha)

            sym = np.digitize(idx, state_loc) - 1
            seq.append(sym)

            temp_alpha = long_alpha[state_loc[sym] : state_loc[sym + 1]]
            temp_alpha /= temp_alpha.sum()
            alpha = temp_alpha

        return seq

    def bridge(self, state1, state2, max_steps=100):
        Pi_x = np.zeros(self.n_clones.sum(), dtype=self.dtype)
        Pi_x[state1] = 1
        log2_lik, mess_fwd = forward_mp_all(
            self.T.transpose(0, 2, 1), Pi_x, self.Pi_a, self.n_clones, state2, max_steps
        )
        s_a = backtrace_all(self.T, self.Pi_a, self.n_clones, mess_fwd, state2)
        return s_a


def updateCE(CE, E, n_clones, mess_fwd, mess_bwd, x, a):
    timesteps = len(x)
    gamma = mess_fwd * mess_bwd
    norm = gamma.sum(1, keepdims=True)
    norm[norm == 0] = 1
    gamma /= norm
    CE[:] = 0
    for t in range(timesteps):
        CE[:, x[t]] += gamma[t]


def forwardE(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """Log-probability of a sequence, and optionally, messages"""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = T_tr[aij].dot(message)
        message *= E[:, j]
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def backwardE(T, E, n_clones, x, a):
    """Compute backward messages."""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T.dtype.type

    # backward pass
    t = x.shape[0] - 1
    message = np.ones(E.shape[0], dtype)
    message /= message.sum()
    mess_bwd = np.empty((len(x), E.shape[0]), dtype=dtype)
    mess_bwd[t] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, j = (
            a[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        message = T[aij].dot(message * E[:, j])
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        mess_bwd[t] = message
    return mess_bwd


@nb.njit
def updateC(C, T, n_clones, mess_fwd, mess_bwd, x, a):
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    timesteps = len(x)
    C[:] = 0
    for t in range(1, timesteps):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (tm1_start, tm1_stop), (t_start, t_stop) = (
            mess_loc[t - 1 : t + 1],
            mess_loc[t : t + 2],
        )
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        q = (
            mess_fwd[tm1_start:tm1_stop].reshape(-1, 1)
            * T[aij, i_start:i_stop, j_start:j_stop]
            * mess_bwd[t_start:t_stop].reshape(1, -1)
        )
        q /= q.sum()
        C[aij, i_start:i_stop, j_start:j_stop] += q


@nb.njit
def forward(T_tr, Pi, n_clones, x, a, store_messages=False):
    """Log-probability of a sequence, and optionally, messages"""
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    j_start, j_stop = state_loc[j : j + 2]
    message = Pi[j_start:j_stop].copy().astype(dtype)
    
    p_obs = message.sum()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_loc = np.hstack(
            (np.array([0], dtype=n_clones.dtype), n_clones[x])
        ).cumsum()
        mess_fwd = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    for t in range(1, x.shape[0]):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        message = np.ascontiguousarray(T_tr[aij, j_start:j_stop, i_start:i_stop]).dot(
            message
        )
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd[t_start:t_stop] = message
    return log2_lik, mess_fwd


@nb.njit
def backward(T, n_clones, x, a):
    """Compute backward messages."""
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T.dtype.type

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    message = np.ones(n_clones[i], dtype) / n_clones[i]
    message /= message.sum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    mess_bwd = np.empty(mess_loc[-1], dtype)
    t_start, t_stop = mess_loc[t : t + 2]
    mess_bwd[t_start:t_stop] = message
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        message = np.ascontiguousarray(T[aij, i_start:i_stop, j_start:j_stop]).dot(
            message
        )
        p_obs = message.sum()
        assert p_obs > 0
        message /= p_obs
        t_start, t_stop = mess_loc[t : t + 2]
        mess_bwd[t_start:t_stop] = message
    return mess_bwd


@nb.njit
def forward_mp(T_tr, Pi, n_clones, x, a, store_messages=False):
    """Log-probability of a sequence, and optionally, messages"""
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    j_start, j_stop = state_loc[j : j + 2]
    message = Pi[j_start:j_stop].copy().astype(dtype)
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_loc = np.hstack(
            (np.array([0], dtype=n_clones.dtype), n_clones[x])
        ).cumsum()
        mess_fwd = np.empty(mess_loc[-1], dtype=dtype)
        t_start, t_stop = mess_loc[t : t + 2]
        mess_fwd[t_start:t_stop] = message
    else:
        mess_fwd = None

    for t in range(1, x.shape[0]):
        aij, i, j = (
            a[t - 1],
            x[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        (i_start, i_stop), (j_start, j_stop) = (
            state_loc[i : i + 2],
            state_loc[j : j + 2],
        )
        new_message = np.zeros(j_stop - j_start, dtype=dtype)
        for d in range(len(new_message)):
            new_message[d] = (T_tr[aij, j_start + d, i_start:i_stop] * message).max()
        message = new_message
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            t_start, t_stop = mess_loc[t : t + 2]
            mess_fwd[t_start:t_stop] = message
    return log2_lik, mess_fwd


@nb.njit
def rargmax(x):
    # return x.argmax()  # <- favors clustering towards smaller state numbers
    return np.random.choice((x == x.max()).nonzero()[0])


@nb.njit
def backtrace(T, n_clones, x, a, mess_fwd):
    """Compute backward messages."""
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    mess_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones[x])).cumsum()
    code = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    i = x[t]
    t_start, t_stop = mess_loc[t : t + 2]
    belief = mess_fwd[t_start:t_stop]
    code[t] = rargmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij, i, j = (
            a[t],
            x[t],
            x[t + 1],
        )  # at time t -> t+1 we go from observation i to observation j
        (i_start, i_stop), j_start = state_loc[i : i + 2], state_loc[j]
        t_start, t_stop = mess_loc[t : t + 2]
        belief = (
            mess_fwd[t_start:t_stop] * T[aij, i_start:i_stop, j_start + code[t + 1]]
        )
        code[t] = rargmax(belief)
    states = state_loc[x] + code
    return states


def backtraceE(T, E, n_clones, x, a, mess_fwd):
    """Compute backward messages."""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    states = np.zeros(x.shape[0], dtype=np.int64)

    # backward pass
    t = x.shape[0] - 1
    belief = mess_fwd[t]
    states[t] = rargmax(belief)
    for t in range(x.shape[0] - 2, -1, -1):
        aij = a[t]  # at time t -> t+1 we go from observation i to observation j
        belief = mess_fwd[t] * T[aij, :, states[t + 1]]
        states[t] = rargmax(belief)
    return states


def forwardE_mp(T_tr, E, Pi, n_clones, x, a, store_messages=False):
    """Log-probability of a sequence, and optionally, messages"""
    assert (n_clones.sum(), len(n_clones)) == E.shape
    dtype = T_tr.dtype.type

    # forward pass
    t, log2_lik = 0, np.zeros(len(x), dtype)
    j = x[t]
    message = Pi * E[:, j]
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik[0] = np.log2(p_obs)
    if store_messages:
        mess_fwd = np.empty((len(x), E.shape[0]), dtype=dtype)
        mess_fwd[t] = message
    for t in range(1, x.shape[0]):
        aij, j = (
            a[t - 1],
            x[t],
        )  # at time t-1 -> t we go from observation i to observation j
        message = (T_tr[aij] * message.reshape(1, -1)).max(1)
        message *= E[:, j]
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik[t] = np.log2(p_obs)
        if store_messages:
            mess_fwd[t] = message
    if store_messages:
        return log2_lik, mess_fwd
    else:
        return log2_lik


def forward_mp_all(T_tr, Pi_x, Pi_a, n_clones, target_state, max_steps):
    """Log-probability of a sequence, and optionally, messages"""
    # forward pass
    t, log2_lik = 0, []
    message = Pi_x
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik.append(np.log2(p_obs))
    mess_fwd = []
    mess_fwd.append(message)
    T_tr_maxa = (T_tr * Pi_a.reshape(-1, 1, 1)).max(0)
    for t in range(1, max_steps):
        message = (T_tr_maxa * message.reshape(1, -1)).max(1)
        p_obs = message.max()
        assert p_obs > 0
        message /= p_obs
        log2_lik.append(np.log2(p_obs))
        mess_fwd.append(message)
        if message[target_state] > 0:
            break
    else:
        assert False, "Unable to find a bridging path"
    return np.array(log2_lik), np.array(mess_fwd)


def backtrace_all(T, Pi_a, n_clones, mess_fwd, target_state):
    """Compute backward messages."""
    states = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    actions = np.zeros(mess_fwd.shape[0], dtype=np.int64)
    n_states = T.shape[1]
    # backward pass
    t = mess_fwd.shape[0] - 1
    actions[t], states[t] = (
        -1,
        target_state,
    )  # last actions is irrelevant, use an invalid value
    for t in range(mess_fwd.shape[0] - 2, -1, -1):
        belief = (
            mess_fwd[t].reshape(1, -1) * T[:, :, states[t + 1]] * Pi_a.reshape(-1, 1)
        )
        a_s = rargmax(belief.flatten())
        actions[t], states[t] = a_s // n_states, a_s % n_states
    return actions, states



def add_death(mat, eps=1e-15):
    # Given an illigal non-stay action, there is eps chance to die
    for action in range(mat.shape[-1]):
        zer = np.where(mat[..., action].sum(axis=0) == 0)[0]
        mat[-1, zer, action] = eps
    # Stay dead once reached
    mat[-1, -1, :] = 1.0
    return mat


def add_stationary(mat, eps=1e-15):
    """
    Add an action to stand still
    """
    return np.concatenate(
        [mat, np.eye(mat.shape[0]).reshape(*mat[..., :1].shape)], axis=2
    )

def set_stationary(mat, idx=-1):
    mat[:,:,idx] = np.eye(mat.shape[0])
    return mat
def set_T_stationary(mat, idx=-1):
    mat[idx,:,:] = np.eye(mat.shape[-1])
    return mat


def forward_mp_all_multiple_states(
    T_tr, Pi_x, Pi_a, n_clones, target_states, max_steps
):
    """Log-probability of a sequence, and optionally, messages"""
    # forward pass
    t, log2_lik = 0, []
    message = Pi_x
    p_obs = message.max()
    assert p_obs > 0
    message /= p_obs
    log2_lik.append(np.log2(p_obs))
    mess_fwd = []
    mess_fwd.append(message)
    T_tr_maxa = (T_tr * Pi_a.reshape(-1, 1, 1)).max(0)
    selected_state = -1
    for t in range(1, max_steps):
        message = (T_tr_maxa * message.reshape(1, -1)).max(1)
        p_obs = message.max()
        if p_obs > 0:
            message /= p_obs
            log2_lik.append(np.log2(p_obs))
            mess_fwd.append(message)

            break_out = False
            for target_state in target_states:
                if message[target_state] > 0.2 and np.any(message != [message[0]]*len(message)):
                    selected_state = target_state
                    break_out = True
                    break
            if break_out:
                break
        else:
            return False

    else:
        return False
    return np.array(log2_lik), np.array(mess_fwd), selected_state