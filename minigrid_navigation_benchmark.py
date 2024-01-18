import numpy as np
import argparse

from ours.V3_3 import Ours_V3_3 
from ours.V1 import Ours_V1
from cscg.cscg import CHMM, datagen_structured_obs_room

from envs.minigrid import GridWorldEnv
from visualisation_tools import *

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')
parser.add_argument("--env",
        type=str,
        help="choose between minigrid,",
        )
parser.add_argument("--room_choice",
    type=int,
    help="minigrid: int between 1-5 ",
    default= 1
    )
parser.add_argument("--model",
    type=str,
    help="choose between: cscg, ours_v3, ours_v1 . \
        Two can be chosen at once for test 2",
    default= 'ours_v3'
    )

parser.add_argument('-p', #used as python minigrid_navigation_benchmark.py --env minigrid -p 0 -p 1
    '--start_pose', 
    type=int,
    action='append', 
    help='enter row, col poses with a space in between', 
    required=True)

parser.add_argument('--test',
    type=int,
    help="which test to run")

parser.add_argument('--max_steps',
    type=int,
    help="how many steps do we allow the agent to take",
    default = 100)

parser.add_argument('--nav',
        type='str',
        help='explo or goal',
        default='exlpo')

def room_init(room_choice:int = 1):

    if room_choice == 1: #3x3 rooms, 1 ob per room - WT ALIAS
        rooms = np.array(
            [
                [0, 0, 1],
                [2, 0, 4],
                [3, 3, 3],
            ]
        )
    
    elif  room_choice == 2: #3x3 rooms, 1 ob per room - NO ALIAS

        rooms = np.array(
            [
                [0, 1, 2],
                [5, 4, 3],
                [6, 7, 8],
            ]
        ) 

    elif  room_choice == 3: #3x4 rooms, 1 ob per room - WT ALIAS

        rooms = np.array(
            [
                [0, 0, 1, 4],
                [2, 0, 1, 3],
                [3, 3, 3, 0],
            ]
        ) #3x4 rooms, 1 ob per room
    
    elif  room_choice == 4: #4x4 rooms, 1 ob per room - WT ALIAS

        rooms = np.array(
            [
                [0, 0, 1, 4],
                [2, 0, 1, 3],
                [3, 3, 3, 0],
                [1, 0, 4, 0],
            ]
        ) #4x4 rooms, 1 ob per room

    elif  room_choice == 5: #4x4 rooms, 1 ob per room - NO ALIAS

        rooms = np.array(
            [
                [0, 1, 2, 3],
                [7, 6, 5, 4],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ]
        )
    else:
        raise "Room_choice expected between 1 and 5"
    custom_colors = (
        np.array(
            [
                [255, 0, 0],#red
                [0, 255, 0], #green
                [50,50, 255], #bluish 
                [112, 39, 195], #purple
                [255, 255, 0], #yellow
                [100, 100, 100], #grey
                [115, 60, 60], #brown
                [255, 255, 255], #white
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

def set_models(models_names, rooms, actions, ob, start_pose, start_state_idx, nav_type):
    if 'explo' in nav_type:
        utility_term = False
    else:
        utility_term = True
        
    models = [None, None]
    if 'oursv3' in models_names:
            models[models_names.index('oursv3')] = Ours_V3_3(num_obs=2, \
                num_states=2, observations=[ob,start_pose], learning_rate_pB=3.0,\
                 actions= actions, utility_term=utility_term)
    if 'oursv1' in models_names:
            models[models_names.index('oursv1')]  = Ours_V1(num_obs=2, \
                num_states=2, observations=[ob,start_state_idx], learning_rate_pB=3.0,\
                actions= actions, utility_term=utility_term)
    if 'cscg' in models_names:
        n_emissions = rooms.max() + 1
        n_clones = np.ones(n_emissions, dtype=np.int64) * 5
        n_actions = max(list(actions.values()))
        x = a = np.array([n_actions])
        models[models_names.index('cscg')]  =  CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=42) 
        models[models_names.index('cscg')].pseudocount = 1.0
    model, model2 = models
    return model, model2

def define_perfect_B(env,rooms, actions):
    """ The perfect B is defined as B[next_state, prev_state, action]"""
    #perfect B for this room config
    desired_state_mapping = {(i * rooms.shape[1] + j): (i, j) for i in range(rooms.shape[0]) for j in range(rooms.shape[1])}
    P = {}
    dim = rooms.shape
    for state_index, xy_coordinates in desired_state_mapping.items():
        P[state_index] = {a : [] for a in range(len(actions))}
        for action in actions.values():
            pose = env.next_p_given_a_known_env(xy_coordinates, action)
            #print('action', action, 'state coordinates', state_index, xy_coordinates, 'next pose', pose)
            next_state_idx = next(key for key, value in desired_state_mapping.items() if value == pose)
            P[state_index][action] = next_state_idx


    num_states = len(desired_state_mapping)
    B = np.zeros([num_states, num_states, len(actions)])
    # print(B.shape)
    for s in range(num_states):
        # print('s', s, perfect_state_mapping[s])
        for a in range(len(actions)):
            ns = int(P[s][a])
            # print('ps', s, 'a', a, 'ns',ns)
            B[ns, s, a] = 1
    return B, desired_state_mapping

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

def main(flags):
    print('receieved args:',flags)
    rooms, cmap = room_init(flags.room_choice)
    
    pose = tuple(flags.start_pose)
    if flags.env == 'minigrid':
        actions = {'UP':2, 'RIGHT':1, 'DOWN':3, 'LEFT':0, 'STAY':4}
        env = GridWorldEnv(rooms, actions)

    perfect_B, desired_state_mapping = define_perfect_B(env,rooms, actions)

    ob = env.reset(pose)
    start_state_idx = env.get_state(pose)
    models_names = flags.model.lower().split(',')
    assert len(models_names) <= 2 , "Maximum 2 models separated by a comma"
    model, model2 = set_models(models_names, rooms, actions, ob, pose, start_state_idx)

    action_hist = np.array([], dtype=int)
    observations = np.array([[int(ob),pose]], dtype=object)
    next_possible_actions = env.get_next_possible_motions(pose)
    for t in range(0, flags.max_steps):
        
        action = model.infer_action(next_possible_actions)
         
        if flags.test == 3 and 'cscg' in models_names[0] and t > 0:
            visual_ob_hist = np.array(observations[1:t,0], dtype=np.int64)
            progression = model.learn_em_T(visual_ob_hist, action_hist[:t], n_iter=200)  # Training
            model.learn_viterbi_T(visual_ob_hist, action_hist[:t], n_iter=1)
    
        ob, pose = env.step(action, pose)
        action_hist = np.append(action_hist,action)
        observations =  np.concatenate([observations, [[int(ob), pose]]], 0) 

        next_possible_actions = env.get_next_possible_motions(pose)
        print('timestep',t,',action taken:',int(action),'pose:',pose,',ob:', ob)
        model.agent_step_update(action,[ob,pose],next_possible_actions)

        if 'cscg' not in models_names[0]:
            if agent_B_match_ideal_B_v2(model.get_B(), perfect_B, model.get_agent_state_mapping(), \
                                     desired_state_mapping, actions, tolerance_margin= 0.4):
                print('Transition matrix is good')
                break
    
    visual_ob_hist = np.array(observations[1:,0], dtype=np.int64)
    if 'cscg' in models_names[0] :
        # print('x', visual_ob_hist, visual_ob_hist.shape, type(visual_ob_hist), type(visual_ob_hist))
        # print('a', action_hist, action_hist.shape, type(action_hist), type(action_hist[0]))
        if flags.test == 3 :
            graph = plot_cscg_graph(
            model, visual_ob_hist, action_hist, specific_str='room:'+str(flags.room_choice) +'_'+ models_names[0]+"_step_per_step", cmap=cmap
            )
        
        progression = model.learn_em_T(visual_ob_hist, action_hist, n_iter=200)
        # refine learning
        model.pseudocount = 1.0
        model.learn_viterbi_T(visual_ob_hist, action_hist, n_iter=1)
        graph = plot_cscg_graph(
            model, visual_ob_hist, action_hist, specific_str='room:'+str(flags.room_choice) +'_'+ models_names[0]+"_seq_a", cmap=cmap
            )
    else:
        graph = plot_graph_as_cscg(
            model.agent.B, model.agent_state_mapping, cmap=cmap, specific_str='room:'+str(flags.room_choice) +'_'+ models_names[0], edge_threshold=0.05
            )
    agent_state_mapping = model.get_agent_state_mapping(visual_ob_hist,action_hist, observations[1:,1])
    plot_transition_detailed_resized(model.get_B(), actions, model.get_n_states(),agent_state_mapping, \
                             desired_state_mapping, models_names[0], plot=False, save=True)

    plot_map(rooms, cmap, show = True)
    plot_path_in_map(env, tuple(flags.start_pose), action_hist, cmap)
if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)