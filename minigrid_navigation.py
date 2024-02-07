import numpy as np
from visualisation_tools import B_to_ideal_B, get_frame
import traceback



def minigrid_exploration(env, model, actions, model_name, pose, max_steps, stop_condition = None, given_policy=None):
    """ 
    
    """
    agent_infos = []
    actions = []
    p_obs = [pose]
    ob = env.get_ob_given_p(pose)
    c_obs = [ob]
    stop = False
    error_message = False
    perfect_B, desired_state_mapping = env.define_perfect_B()
    frames = [get_frame(env, pose)]
    if given_policy is not None:
        max_steps = len(given_policy)-1
        
    if 'pose' in model_name:
        observation = [ob, pose]
    else:
        observation = [ob]

    for t in range(1,max_steps+1):

        try:
            if given_policy is None:
                if 'ours' in model_name :
                    action, agent_info = ours_action_decision(model)
                if 'cscg' in model_name:
                    if 'random' in model_name:
                        random_policy = True
                    else:
                        random_policy = False
                    
                    action, agent_info = cscg_action_decision(model, observation, env.get_possible_motions(), random_policy)
            else:
                action = given_policy[t]
                agent_info = update_model_given_action(model, action, pose)

            obs, _,_,_ = env.step(action, pose)
            ob, pose = obs
            frames.append(get_frame(env, pose))
            next_possible_actions = env.get_next_possible_motions(pose, no_stay = True)
            if 'pose' in model_name:
                if pose not in model.pose_mapping: #this is just a security check
                    model.pose_mapping.append(pose)
                observation = [ob, pose]
            else:
                observation = [ob]

            if 'ours' in model_name:
                ours_update_agent(model, action, observation, next_possible_actions)

            #Save data
            actions.append(action)
            c_obs.append(ob)
            p_obs.append(pose)
            agent_infos.append(agent_info)

            #Verify if goal reached (if desired)
            if 'explo_done' in stop_condition:
                stop = explo_reached(model, perfect_B, desired_state_mapping, env.possible_actions, tolerance_margin = 0.4)
            
            if stop :
                print('Transition matrix is good')
                break
        except :
            error_message = traceback.format_exc() 
            print('ERROR MESSAGE:',error_message)
            print('EXPERIMENT INTERRUPTED')  
            break
        
    #We want the same amount of actions than observations.
    actions = np.insert(actions, 0, -1)
    data = {
        "steps": [*range(t+1)],
        "c_obs": c_obs,
        "actions": actions,
        "poses": p_obs,
        'stop_condition_' + str(stop_condition): stop,
        "agent_info": agent_infos,
        "frames":frames,
        "error":error_message,
    }
    if 'cscg' in model_name:
        if 'pose' in model_name:
            observations = np.array([np.array([c, p], dtype=object) for c, p in zip(c_obs, p_obs)])
        
        model, train_progression = train_cscg(model, observations, actions[1:])
        data['train_progression'] = train_progression
    
    
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
    for t in range(1,max_steps+1):
        try:
            if 'ours' in model_name :
                action, agent_info = ours_action_decision(model, [ob, pose], next_possible_actions)
            if 'cscg' in model_name:
                action, agent_info = cscg_action_decision(model, ob, next_possible_actions)
                
            obs, _,_,_ = env.step(action, pose)
            ob, pose = obs
            next_possible_actions = env.get_next_possible_motions(pose, no_stay = False)
            frames.append(get_frame(env, pose))
            if 'ours' in model_name:
                ours_update_agent(model, action, [ob,pose], next_possible_actions)

            #Save data
            actions.append(action)
            c_obs.append(ob)
            p_obs.append(pose)
            agent_infos.append(agent_info)

            #Verify if goal reached (if desired)
            if 'goal_reached' in stop_condition:
                stop = goal_reached(model, action , c_obs, p_obs, actions_dict)
            
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
def explo_reached(model, perfect_B, desired_state_mapping, actions, tolerance_margin = 0.4):
    return agent_B_match_ideal_B_v2(model, perfect_B, 
            desired_state_mapping, actions, tolerance_margin= tolerance_margin)
           

def agent_B_match_ideal_B_v2(model, perfect_B, desired_state_mapping, actions, tolerance_margin = 0.3):
    """Check if the values == 1 in perfect_B are filled with values relatively close at tolerance level"""
    agent_state_mapping = model.get_agent_state_mapping()
    agent_B = model.get_B()
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
    if isinstance(observations[0], np.ndarray) :
        poses_idx = model.from_pose_to_idx(observations[:,1])
        observations[:,1] = poses_idx
    elif isinstance(observations[0], tuple):
        poses_idx = model.from_pose_to_idx(observations)
        observations = poses_idx
    
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
    return {'qs': model.get_current_belief()[0], "bayesian_surprise":0}
#======================= GOAL METHODS ===================================#

def goal_reached(model, action , c_obs, p_obs, actions):
    if (c_obs[-1] == model.preferred_ob[0] and action == actions['STAY']) :
        print('Goal reached')
        return 1
    elif np.array_equal([model.preferred_ob[0]] * 3,c_obs[-3:]) and np.array_equal([p_obs[-1]]*3 ,p_obs[-3:]):
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
        
def ours_action_decision(ours, observation=None, next_possible_actions=None):
    """ 
    Our model infer an action between all actions and update its believes 
    according to how he moved and where it can move at the next step.
    """
    action, agent_info = ours.infer_action(observation=observation, next_possible_actions=next_possible_actions)
    return action, agent_info

def ours_update_agent(ours, action, observations, next_possible_actions):
    ours.agent_step_update(action,observations,next_possible_actions)

