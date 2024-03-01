import argparse
import numpy as np
import pickle
from pathlib import Path
import os 
import sys
import pandas as pd
from datetime import datetime
from envs.minigrid import GridWorldEnv
from minigrid_navigation import minigrid_reach_goal, minigrid_exploration
from ours.V3_3 import Ours_V3_3 
from ours.V1 import Ours_V1
from ours.V4 import Ours_V4
from ours.V4_2 import Ours_V4_2
from cscg.cscg import CHMM
from visualisation_tools import generate_plot_report, generate_csv_report, save_transitions_plots
import traceback

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')
parser.add_argument("--env",
        type=str,
        help="choose between grid_3x3_alias, grid_3x4 etc",
        default= 'grid_4x4'
        )
parser.add_argument("--model",
    type=str,
    help="choose between: cscg, ours_v3, ours_v1, cscg_random_policy",
    default= 'ours_v3'
    )
parser.add_argument("--load_model",
    type=str,
    help="give a pickle path",
    default = 'None'
    )
parser.add_argument('-p', #used as python minigrid_navigation_benchmark.py --env minigrid -p 0 -p 1
    '--start_pose', 
    type=int,
    action='append', 
    help='enter row, col poses with a space in between', 
    required=True)

parser.add_argument('--goal',
    type=int,
    help="Which observation do we want to reach?",
    default = -1)

parser.add_argument('--max_steps',
    type=int,
    help="how many steps do we allow the agent to take before stopping experiment",
    default = 100)

parser.add_argument('--stop_condition',
    type=str,
    help="explo_done or goal_recahed can be used to stop experiment before max_step reached",
    default = 'None')

parser.add_argument('--load_policy',
    type=str,
    help="path to a csv file containing a policy",
    default = 'None')

parser.add_argument('--inf_algo',
    type=str,
    help="Inference algo Vanilla or MMP",
    default = 'VANILLA')


def find_directory(directory_name):
    start_dir = Path.cwd() / 'results'
    for root, dirs, files in os.walk(start_dir):
        if directory_name in dirs:
            return os.path.join(root, directory_name)
        
def load_object(load_path):
    if not os.path.exists(load_path):
        #If the path does not exist, let's try that?
        load_path = find_directory(load_path)
        if load_path[-4:] != '.pkl':
            for file in os.listdir(load_path):
                if file.endswith(".pkl"):
                    load_path = os.path.join(load_path, file) 
            
    with open(load_path, "rb") as inp:
        agent = pickle.load(inp)
    return agent

def load_a_policy(policy_path):
    import ast
    if not os.path.exists(policy_path):
        #If the path does not exist, let's try that?
        policy_path = find_directory(policy_path)
        if policy_path[-5:] != '.xlsx':
            policy_path = os.path.join(policy_path, 'logs.xlsx')
        
    
    excel_data = pd.read_excel(policy_path, converters={"poses": ast.literal_eval})
    policy = excel_data['actions'].tolist()
    start_pose = excel_data['poses'].tolist()[0]
    return policy, start_pose

def dump_object(model, model_name, save_name):
    with open(str(save_name) + '/'+ model_name +'.pkl', "wb") as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)


def create_store_path(model_name, env_name):
    cwd = Path.cwd()
    if 'cscg' in model_name :
        name = 'cscg'
    else:
        name = 'ours'
    if 'goal' in model_name:
        model_name_prefix = model_name.split('_goal_ob')
        nav_type = name+ '_goal/' + model_name_prefix[0]
    else:
        nav_type = name +'_exploration/' + model_name
    
    dp = cwd / 'results'/ env_name / nav_type
    # elif 'ours' in model_name and 'goal' in model_name:
    #     dp = cwd / 'results'/ env_name / 'ours_goal' 
    # else:
    #     dp = cwd / 'results'/ env_name / model_name
    day = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H-%M-%S")
    save_name = str(model_name)+ '_' + str(day) + '-'+ str(now)
    
    store_path = dp / save_name

    store_path.mkdir(exist_ok=True, parents=True)
    return store_path.resolve()

def set_models(possible_actions:dict, rooms:np.ndarray, \
               obs_c_p:list, start_state:int, flags):
    
    model_name = flags.model

    if 'pose' in model_name:
        observation = obs_c_p
        n_emissions = rooms.shape[0] * rooms.shape[1]
        ambiguity = False
    else:
        observation = [obs_c_p[0]]
        n_emissions = rooms.max() + 1
        ambiguity = True

    if 'ours_v3' in model_name:
            model = Ours_V3_3(num_obs=2, \
                num_states=2, observations=observation, \
                    learning_rate_pB=3.0, actions= possible_actions) #dim is still 2 set as default
    elif 'ours_v4_2' in model_name: #LOOKAHEAD : 8
            model = Ours_V4_2(num_obs=2, \
                num_states=2, observations=observation, \
                    learning_rate_pB=3.0, actions= possible_actions, lookahead=5, inference_algo = flags.inf_algo)
    elif 'ours_v4' in model_name:
            model = Ours_V4(num_obs=2, \
                num_states=2, observations=observation, \
                    learning_rate_pB=3.0, actions= possible_actions, inference_algo = flags.inf_algo)
    
            
    elif 'ours_v1' in model_name:
            model = Ours_V1(num_obs=2, \
                num_states=2, observations=[observation[0],start_state], \
                learning_rate_pB=3.0, actions= possible_actions)

    elif 'cscg' in model_name:
        n_clones = np.ones(n_emissions, dtype=np.int64) * 10
        n_actions = max(list(possible_actions.values()))
        x = np.array([0])
        a = np.array([n_actions])
        model = CHMM(n_clones=n_clones, pseudocount=0.05, \
                       x=x, a=a, possible_actions=possible_actions, seed=flags.seed, \
                        set_stationary_B=True, ob_ambiguity = ambiguity) 
    else:
        raise ValueError(str(model_name) + ' is not a recognised model name')
    
    return model

def main(flags):
        
    available_models = ['cscg_random_policy', 'cscg', 'ours_v3', 'ours_v5', 'ours_v4']
    data = {}
    try:
        #SETUP ENVIRONMENT
        env_name = flags.env
        if 'grid' in env_name:
            possible_actions = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'STAY':4}
            env = GridWorldEnv(env_name, possible_actions,\
                            max_steps=flags.max_steps, goal=flags.goal)
            perfect_B, desired_state_mapping = env.define_perfect_B()
        
            #adapt policy type (given or not)
            if flags.load_policy != 'None':
                policy, pose = load_a_policy(flags.load_policy)
                flags.model+='_given_policy'
            else:
                policy= None
                pose = tuple(flags.start_pose)
            try:
                obs_c_p,_ = env.reset(pose)
                if obs_c_p[0] == -1:
                    print('Terminating experiment, we are starting in a wall')
                    return
            except IndexError: #We gave a starting pose that is not in env
                print('Terminating experiment, we are starting outside the env range')
                return 
            
            state = env.get_state(pose)
            
            #SET MODEL
            if flags.load_model != 'None' :
                print('Loading model from: ', flags.load_model)
                model = load_object(flags.load_model)
                model_path = flags.load_model.split('/')[::-1]
                model_name = next((item for item in model_path if any(model in item for model in available_models)), None)
                model.current_pose = None
                if hasattr(model, 'agent'):
                    model.agent.reset()
                else:
                    model.reset()
                                
                #model_name = [substring for substring in available_models if substring == flags.load_model][0]
            else:
                print('Creating model: ', flags.model)
                flags.seed = np.random.randint(low=10000)
                model = set_models(possible_actions, env.rooms, obs_c_p, state, flags)
                model_name = flags.model
                if 'ours_v4' in model_name:
                    model_name+= '_'+ flags.inf_algo
                    #TODO: REMOVE AFTER TESTS
                    model.test_display_policy_imagination = True #TEMPORARY FOR TEST PURPOSES
                    model.sampling_mode=  'marginal'
                    model.action_selection = 'deterministic'

            print('model_name', model_name)
            
                            
            #SAVING TERMINAL LOGS
            if flags.goal >= 0:
                model_name+='_goal_ob:'+str(flags.goal)
            store_path = create_store_path(model_name, env_name)
            
            txt_file = str(store_path) + '/'+ str(flags.model) +"_SP:"+ str(flags.start_pose) +"_G:"+ str(flags.goal) +".txt"
            print('Saving terminal prints in txt file:', txt_file)
            txt_terminal_prints = open(txt_file, 'a+')
            sys.stdout = txt_terminal_prints
            
            #SET NAVIGATION TYPE AND RUN NAVIGATION 
            if flags.goal >= 0:
                print('SEARCHING GOAL')
                                
                output = env.get_short_term_goal()
                print('goal_ob', env.goal_ob, output[2])
                if output != None:

                    preferred_ob = [flags.goal, -1] # [c_ob, pose or state]
                    model.goal_oriented_navigation(preferred_ob, inf_algo=flags.inf_algo)
                    if 'ours' in model_name and flags.load_model != 'None':
                        model.reset()
                    
                    data = minigrid_reach_goal(env, model, possible_actions, model_name, pose, \
                                            flags.max_steps, stop_condition = flags.stop_condition.lower())
                else:
                    data = {
                        "steps": 0,
                        "c_obs": obs_c_p[0],
                        "actions": -1,
                        "poses": obs_c_p[1],
                        'stop_condition_NO_GOAL_FOUND_IN_ENV': 0,
                        "agent_info": {},
                        "frames":[],
                        "error": 'Goal_not_in_env',
                    }      
            else:
                print('STARTING EXPLO')
                store_path = create_store_path(model_name, env_name)
                model.explo_oriented_navigation(inference_algo=flags.inf_algo)
                model, data = minigrid_exploration(env, model, model_name, pose, flags.max_steps, \
                                                     stop_condition = flags.stop_condition.lower(), \
                                                    given_policy=policy)
                
                dump_object(model, model_name, store_path)
                
        else:
            raise "Only implemented Minigrid testbench"
    except:
        error_message = traceback.format_exc() 
        print('ERROR MESSAGE:',error_message)


    print('Storing values at: ',store_path)
    
    generate_csv_report(data.copy(), flags, store_path)
    generate_plot_report(data, env, store_path)
    save_transitions_plots(model, model_name, possible_actions, desired_state_mapping, data, env.rooms_colour_map, store_path)
    
    print('Experiment done')
    
    txt_terminal_prints.close()
    
    

if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)