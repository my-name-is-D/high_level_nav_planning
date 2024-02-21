import numpy as np
from visualisation_tools import B_to_ideal_B, get_frame, from_policy_to_pose
import traceback
import copy
import gc

def minigrid_exploration(env, model, model_name, pose, max_steps, stop_condition = None, given_policy=None):
    """ 
    
    """
    data = {}
    agent_infos = []
    actions = []
    p_obs = [pose]
    ob = env.get_ob_given_p(pose)
    c_obs = [ob]
    stop = False
    error_message = False
 
    perfect_B, desired_state_mapping = env.define_perfect_B()
    init_model = []

    frames = [get_frame(env, pose)]
    if given_policy is not None:
        max_steps = len(given_policy)-1
        
    if 'pose' in model_name:
        observation = [ob, pose]
        real_pose_dict = None
        
    else:
        observation = [ob]
        real_pose_dict = {model.current_pose: pose}
        
    next_possible_actions = env.get_next_possible_motions(pose, no_stay = False)

    for t in range(1,max_steps+1):
        print('step:', t)
        try:
            if given_policy is None:
                if 'ours' in model_name :
                    action, agent_info = ours_action_decision(model, next_possible_actions=next_possible_actions)
                    ours_infer_position(model, observation, action, next_possible_actions)
                elif 'cscg' in model_name:
                    if 'random' in model_name:
                        random_policy = True
                    else:
                        random_policy = False
                    #NOTE: Both can be given interchangably, the diff is in the Transition matrix as giving only possible motions
                    # means it will never experiment walls impact on states.
                    motions = env.get_possible_motions() #next_possible_actions 
                    if init_model != []:
                        action, agent_info = cscg_action_decision(init_model, observation,motions , random_policy)
                        model.agent_state_mapping = init_model.get_agent_state_mapping()
                        model.pose_mapping = init_model.get_pose_mapping()
                    else:
                        action, agent_info = cscg_action_decision(model, observation,motions , random_policy)

                   
            else:
                action = given_policy[t]
                agent_info = update_model_given_action(model, action, pose)

            obs, _,_,_ = env.step(action, pose)
            ob, pose = obs
            
            next_possible_actions = env.get_next_possible_motions(pose, no_stay = False)
            if 'pose' in model_name:
                # if pose not in model.pose_mapping: #this is just a security check
                #     model.pose_mapping.append(pose)
                observation = [ob, pose]
            else:
                observation = [ob]
                

            if 'ours' in model_name:
                ours_update_agent(model, action, observation, next_possible_actions)
                if real_pose_dict is not None:
                    #print('action:',action,'model pose:',model.current_pose,'current env pose:', pose)
                    real_pose_dict[model.current_pose] = pose 
            elif 'cscg' in model_name:
                if t % 5 == 0:
                    if 'pose' in model_name:
                        observations = np.array([np.array([c, p], dtype=object) for c, p in zip(c_obs[:len(actions)], p_obs[:len(actions)])])
                    else:
                        observations = c_obs
                        
                    del init_model
                    gc.collect()
                    init_model = copy.deepcopy(model)
                    print(len(observations), len(actions))
                    init_model, train_progression = train_cscg(init_model, observations, actions)
                    data['train_progression'] = train_progression
                    if isinstance(observations[0], np.ndarray) or isinstance(observations[0], tuple):
                        observations[:,1] = init_model.from_pose_to_idx(observations[:,1])
                    init_model.get_agent_state_mapping(observations,actions,p_obs[:len(actions)])
                    
            #Save data
            frames.append(get_frame(env, pose))
            actions.append(action)
            c_obs.append(ob)
            p_obs.append(pose)
            agent_infos.append(agent_info)

            #Verify if goal reached (if desired)
            
            if 'ours' in model_name:
                stop = transition_explo_reached(model, perfect_B, desired_state_mapping, env.possible_actions, \
                                    tolerance_margin = 0.4, real_pose_dict = real_pose_dict)
            elif 'cscg' in model_name and init_model != []:
                stop = transition_explo_reached(init_model, perfect_B, desired_state_mapping, env.possible_actions, \
                                    tolerance_margin = 0.4, real_pose_dict = None)
            
            if stop :
                if 'cscg' in model_name:
                    model = init_model
                print('Transition matrix is good')
                break
        except :
            error_message = traceback.format_exc() 
            print('ERROR MESSAGE:',error_message)
            print('EXPERIMENT INTERRUPTED')  
            break
        
    #We want the same amount of actions than observations.
    actions = np.insert(actions, 0, -1)
    data["steps"] = [*range(t+1)]
    data["c_obs"] = c_obs
    data["actions"] = actions
    data["poses"] = p_obs
    data['stop_condition_' + str(stop_condition)] = stop
    data["agent_info"] = agent_infos
    data["frames"] = frames
    data["error"] = error_message
    if 'cscg' in model_name and not stop:
        if 'pose' in model_name:
            observations = np.array([np.array([c, p], dtype=object) for c, p in zip(c_obs, p_obs)])
        
        model, train_progression = train_cscg(model, observations, actions[1:])
        data['train_progression'] = train_progression
    
    #This is for visualisation purposes
    elif real_pose_dict is not None and 'ours' in model_name:
        agent_state_mapping = model.get_agent_state_mapping()
        model.agent_state_mapping = {real_pose_dict[key]: value for key, value in list(agent_state_mapping.items()) if key in real_pose_dict}
    return model, data

def minigrid_reach_goal(env, model, actions_dict, model_name, pose, max_steps, stop_condition = None):
    """
    
    """
    agent_infos = []
    actions = []
    
    p_obs = [pose]
    ob = env.get_ob_given_p(pose)
    c_obs = [ob]
    stop = 0
    frames = [get_frame(env, pose)]
    error_message = False
    next_possible_actions = env.get_next_possible_motions(pose, no_stay = False)

    if 'pose' in model_name:
        observation = [ob, pose]
    else:
        observation = [ob]

    for t in range(1,max_steps+1):
        print('step:', t)
        try:
            if 'ours' in model_name :
                action, agent_info = ours_action_decision(model, observation, next_possible_actions)
                ours_infer_position(model, observation, action, next_possible_actions)
            if 'cscg' in model_name:
                action, agent_info = cscg_action_decision(model, observation, next_possible_actions)
                
            obs, _,_,_ = env.step(action, pose)
            print('action', action, 'colour', obs[0], 'pose', obs[1])
            ob, pose = obs
            next_possible_actions = env.get_next_possible_motions(pose, no_stay = False)
            
            if 'pose' in model_name:
                # if pose not in model.pose_mapping: #this is just a security check
                #     model.pose_mapping.append(pose)
                observation = [ob, pose]
            else:
                observation = [ob]
            
            if 'ours' in model_name:
                ours_update_agent(model, action, observation, next_possible_actions)

            #Save data
            frames.append(get_frame(env, pose))
            actions.append(action)
            c_obs.append(ob)
            p_obs.append(pose)
            agent_infos.append(agent_info)

            #Verify if goal reached (if desired)
            if 'goal_reached' in stop_condition:
                if 'cscg' in model_name and 'pose' in model_name:
                    obs = model.from_obs_to_ob(np.array([np.array([c, p], dtype=object) for c, p in zip(c_obs, p_obs)]))
                else:
                    obs = c_obs
                stop = goal_reached(model, action , obs, p_obs, actions_dict)
            
            if stop > 0:
                break
        except:
            error_message = traceback.format_exc() 
            print('ERROR MESSAGE:',error_message)
            print('EXPERIMENT INTERRUPTED')  
            break
    #We want the same amount of actions than observations.
    actions = np.insert(actions, 0, -1)
    return  {
        "steps": [*range(t+1)],
        "c_obs": c_obs,
        "actions": actions,
        "poses": p_obs,
        'stop_condition_' + str(stop_condition): stop,
        "agent_info": agent_infos,
        "frames":frames,
        "error":error_message,
    }


#======================= EXPLO METHODS ===================================#
def transition_explo_reached(model, perfect_B, desired_state_mapping, actions, tolerance_margin = 0.4, real_pose_dict=None):
    agent_state_mapping = model.get_agent_state_mapping()
    agent_B = model.get_B()
    if real_pose_dict is not None:
        agent_state_mapping = {real_pose_dict[key]: value for key, value in list(agent_state_mapping.items()) if key in real_pose_dict}
    return agent_B_match_ideal_B_v2(agent_state_mapping, agent_B, perfect_B, 
            desired_state_mapping, actions, tolerance_margin= tolerance_margin)

def state_explo_reached(model, perfect_B, desired_n_state):

    if model == []:
        return False
    states = model.states
    v = np.unique(states)
    T = model.T[:, v][:, :, v]
    A = T.sum(0)
    A = A.round(3)
    non_zero_mask = perfect_B > 0
    current_n_zero_mask = A >0
    return len(v) == desired_n_state and np.array_equal(non_zero_mask,current_n_zero_mask)
           
def agent_B_match_ideal_B_v2(agent_state_mapping, agent_B, perfect_B, desired_state_mapping, actions, tolerance_margin = 0.3):
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

def train_cscg(model, observations, actions):
    #len(observations) > len(actions), no matter because this algo considers only actions length 
    if isinstance(observations[0], np.ndarray) or isinstance(observations[0], tuple):
        observations = model.from_obs_to_ob(observations)
    
    actions = np.array(actions).flatten().astype(np.int64)

    progression = model.learn_em_T(observations, actions, n_iter=200)  # Training
    # # refine learning
    model.pseudocount = 0.0001
    model.learn_viterbi_T(observations, actions, n_iter=100)
    return model, progression

def update_model_given_action(model:object, action:int, pose:tuple):
    model.agent.action = np.array([action])
    model.agent.step_time()
    if pose not in model.pose_mapping:
        model.pose_mapping.append(pose)
    return {'qs': model.get_belief_over_states()[0], "bayesian_surprise":0}

def define_perfect_cscg(env, model, actions, start_pose):
    """ We have a perfect Transition matrix between the correct number of states"""
    test_model = copy.deepcopy(model) 
    a_test = []
    #NOTE: THIS ASSUME THAT 1000 STEPS ARE ENOUGH IN ANY ENV
    for _ in range(1000):
        a_test.append(np.random.choice(list(actions.values())))
    poses, c_obs = from_policy_to_pose(env, start_pose, a_test, add_rand=False)
    poses = [tuple(map(int, sub)) for sub in poses]
    obss = np.array([np.array([c, p], dtype=object) for c, p in zip(c_obs, poses)])
    test_model.set_agent_state_mapping(obss)

    test_model, _ = train_cscg(test_model, obss.copy(), a_test)
    v = np.unique(test_model.states)
    T = test_model.T[:, v][:, :, v]
    perfect_B = T.sum(0).round(3)
    return perfect_B, len(v)

#======================= GOAL METHODS ===================================#

def goal_reached(model, action , c_obs, p_obs, actions):
    if c_obs[-1] in model.preferred_ob:
        preferred_ob = c_obs[-1]
    else:
        preferred_ob = model.preferred_ob[0]

    if (c_obs[-1] == preferred_ob and action == actions['STAY']) :
        print('Goal reached')
        return 1
    elif np.array_equal([preferred_ob] * 3,c_obs[-3:]) and np.array_equal([p_obs[-1]]*3 ,p_obs[-3:]):
        print('Goal reached')
        return 2
    return 0

        
def cscg_action_decision(cscg, observation, next_possible_actions, random_policy=False):
    """ 
    CSCG model infers his current state using an inferring agent (same as ours)
    then infer the most likely path using viterbi algo.
    """
    action, agent_info = cscg.infer_action(next_possible_actions = next_possible_actions, \
                                    observation = observation, random_policy=random_policy)
                            
    return action, agent_info
        
def ours_action_decision(ours, observation:list=None, next_possible_actions:list=None):
    """ 
    Our model infer an action between all actions and update its believes 
    according to how he moved and where it can move at the next step.
    """
    action, agent_info = ours.infer_action(observation=observation, next_possible_actions=next_possible_actions)
    return action, agent_info

def ours_infer_position(ours, observation:list, action:int, next_possible_actions:list):
    ours.initialise_current_pose(observation)
    ours.infer_pose(action,next_possible_actions)

def ours_update_agent(ours, action, observations, next_possible_actions):
    
    ours.agent_step_update(action,observations,next_possible_actions)

