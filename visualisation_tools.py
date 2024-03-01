from __future__ import annotations
from scipy import special
from typing import Optional, Union
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import igraph
from io import BytesIO
import copy
from matplotlib import cm, colors
import matplotlib.collections as mcoll
import matplotlib.path as mpath

# import io
import imageio
import bisect
import os

def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr

#TODO: Recheck if this is not erasing errors
def set_observation_as_rooms_idx(agent_state_mapping, rooms):
    state_mapping = copy.deepcopy(agent_state_mapping)
    for row in range(rooms.shape[0]):
        for col in range(rooms.shape[1]):
            state_mapping[(row,col)]['ob'] = rooms[row,col]
    return state_mapping

#==== BASIC PLOT PRESENTATION METHODS ====#
def create_custom_cmap(custom_colors):
    return colors.ListedColormap(custom_colors[:])

def colorline(
    x, y, z=None, cmap=plt.get_cmap('Greys'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0, ax= None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


#==== SAVING PLOTS ====#

def save_transitions_plots(model, model_name, actions, desired_state_mapping, run_logs, cmap,store_path):
    if 'pose' in model_name:
        poses_idx = model.from_pose_to_idx(run_logs['poses'])
        observations = np.array([np.array([c, p], dtype=object) for c, p in zip(run_logs['c_obs'], poses_idx)])
    else:
        observations = run_logs['c_obs']
    agent_state_mapping = model.get_agent_state_mapping(observations,run_logs['actions'][1:], run_logs['poses'])
    Transition_matrix = model.get_B()
    T_plot = plot_transition_detailed_resized(Transition_matrix, actions, agent_state_mapping, desired_state_mapping, plot=False)
    plt.savefig(store_path / 'model_transitions.jpg')
    plt.close()

    if 'pose' in model_name:
        poses_idx = model.from_pose_to_idx(run_logs['poses'])
        observations = np.array([np.array([c, p], dtype=object) for c, p in zip(run_logs['c_obs'], poses_idx)])
    else:
        observations = run_logs['c_obs']
        
    if 'ours' in model_name:
        v = [value['state'] for value in agent_state_mapping.values()]
        # obs = [value['ob'] for value in agent_state_mapping.values()]
        T = Transition_matrix[v,:][:,v,:]
        A = T.sum(2).round(1)
        div = A.sum(1, keepdims=True)
        A /= (div + 0.0001)
        e_th = 0.1
        while e_th < 0.6:
            edge_threshold = e_th
            e_th*=1.5
            A[A < edge_threshold] = 0
            plot_graph_as_cscg(A, agent_state_mapping, cmap,store_path, edge_threshold= edge_threshold)

    elif 'cscg' in model_name:
        if len(model.states) == 0:
            c_obs = np.array(run_logs['c_obs']).flatten().astype(np.int64)
            a = np.array(run_logs['actions'][1:]).flatten().astype(np.int64)
            states = model.decode(c_obs,a)[1]
        else:
            states = model.states
        v = np.unique(states)
        T = model.C[:, v][:, :, v]
        A = T.sum(0)
        div = A.sum(1, keepdims=True)
        A /= (div + 0.0001)
        
        state_map = model.get_agent_state_mapping(observations,run_logs['actions'][1:],run_logs['poses'])
        plot_cscg_graph(A, observations, v, model.n_clones, state_map, store_path, cmap)

def generate_csv_report(run_logs, flags, store_path):
    if 'frames' in run_logs.keys():
        del run_logs['frames']
    
    if 'efe_frames' in run_logs.keys():
        del run_logs['efe_frames']

    if 'train_progression' in run_logs.keys():
        del run_logs['train_progression']

    # if 'agent_info' in run_logs.keys():
    #     del run_logs['agent_info']

    # del run_logs['stop_condition_none']
    max_list_length = max(len(value) for value in run_logs.values() if (isinstance(value, list) or isinstance(value, np.ndarray)))

    for key, value in run_logs.items():
        if not isinstance(value, list) and not isinstance(value, np.ndarray) :
            run_logs[key] =  [value] * max_list_length
         
        elif len(value) < max_list_length:
            to_add = [value[-1]] * (max_list_length - len(value))
            run_logs[key] =  np.append(value, to_add)
            
    flags_dict = vars(flags)
    del flags_dict['start_pose']
    for key, value in flags_dict.items():
        run_logs[key] = [value] * max_list_length

    run_logs_df = pd.DataFrame.from_dict(run_logs)
    run_logs_df.to_excel(store_path / "logs.xlsx", index=False, engine='openpyxl')   

def generate_plot_report(run_logs, env, store_path):
    ax = plot_path_in_map(env, run_logs['poses'])
    plt.savefig(store_path /"agent_path.jpg")
    plt.clf()

    # Trajectory gif
    
    #with imageio.get_writer(store_path / 'test.gif', mode='I') as writer:
    imgs_path = store_path/ 'imgs'
    imgs_path.mkdir(exist_ok=True, parents=True)
    for i in range(len(run_logs["frames"])):
        img_name = str(i)+'image.jpg'
        imageio.imwrite(imgs_path / img_name, run_logs['frames'][i])
        # image = imageio.imread(store_path /name)
        # writer.append_data(image)

    gif_path = store_path / "navigate.gif"
    imageio.mimsave(gif_path, [imageio.imread(f"{store_path}/imgs/{i}image.jpg") for i in range(len(run_logs['frames']))], 'GIF', duration=0.5, loop=1)
    os.system(f'rm -rf {imgs_path}')
    # with imageio.get_writer(gif_path, mode='I', duration=500) as writer:
    #     # Append each frame to the GIF writer
    #     for frame in run_logs["frames"]:
    #         writer.append_data(frame)
    
    # gif_path = store_path / "navigate.gif"
    # imageio.mimsave(gif_path, run_logs["frames"], 'GIF', duration=500, loop=0)

    #EFE POSES VISUALISATION
    if 'efe_frames' in run_logs:
        imgs_path = store_path/ 'efe_imgs'
        imgs_path.mkdir(exist_ok=True, parents=True)
        for i in range(len(run_logs["efe_frames"])):
            img_name = 'step_'+str(i)+'.jpg'
            imageio.imwrite(imgs_path / img_name, run_logs['efe_frames'][i])
            # image = imageio.imread(store_path /name)
            # writer.append_data(image)

        gif_path = store_path / "efe_frames.gif"
        imageio.mimsave(gif_path, [imageio.imread(f"{store_path}/efe_imgs/step_{i}.jpg") for i in range(len(run_logs['efe_frames']))], 'GIF', duration=0.5, loop=1)

    # Entropy plot
    state_beliefs = [log["qs"] for log in run_logs["agent_info"]]
    entropies = [entropy(s) for s in state_beliefs]
    plt.figure()
    plt.plot(np.arange(len(entropies)), entropies)
    plt.title("Entropy over full state belief")
    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.savefig(store_path /"entropy_plot.jpg")
    plt.clf()

    # Bayesian Surprise
    surprises = [np.nan_to_num(log["bayesian_surprise"]) for log in run_logs["agent_info"]]
    plt.plot(np.arange(len(surprises)), surprises)
    plt.title("Bayesian Surprise")
    plt.xlabel("Time")
    plt.ylabel("Surprise")
    plt.grid(True)
    plt.savefig(store_path /"Bayesian_surprise_plot.jpg")
    plt.clf()

    if 'train_progression' in run_logs:
        ax = plot_progression_T(run_logs['train_progression'])
        plt.savefig(store_path /"train_progression.jpg")

    # Close the Matplotlib plot to release memory (optional)
    plt.close()

#==== AGENT IN ENV PRINT ====#
def plot_observations_and_states(env, pose, agent_state_map=None):
    visualise_position = env.unknown_rooms.copy().astype(object)
    visualise_position[pose[0], pose[1]] = 'x'
    print('position in rooms observations')
    print(visualise_position)
    for position, data in env.state_mapping.items():
        visualise_position[position[0], position[1]] = data['state']
    print('rooms ideal generated states')
    print(visualise_position)
    if agent_state_map is not None:
        for position, data in agent_state_map.items():
            try:
                visualise_position[position[0], position[1]] = data['state']
            except IndexError: #We only consider position in env layout and discard ghost states outside map
                continue
        print('agent generated states')
        print(visualise_position)

#==== MAP PLOTS ====#

def get_efe_frame(env, pose, action_marginals):
    policy_len = action_marginals.shape[0]//2

    fig = plt.figure(1)
    ax = plot_map(env.rooms, cmap=env.rooms_colour_map, show = False, alpha=0.15)

    # Create a mask to apply the filter only to the specified region
    mask = np.zeros_like(env.rooms, dtype=float)

    start_x = max(policy_len - pose[0], 0)
    end_x = min(policy_len + (mask.shape[0]-pose[0]), mask.shape[0]+policy_len)
    start_y = max(policy_len - pose[1], 0)
    end_y = min(policy_len + (mask.shape[1]-pose[1]), mask.shape[1]+policy_len)

    mask_start_x = max(0, pose[0] - policy_len)
    mask_end_x = min(pose[0] + policy_len+1, mask.shape[0])
    mask_start_y = max(0, pose[1] - policy_len)
    mask_end_y = min(pose[1] + policy_len+1, mask.shape[1])

    mask[mask_start_x:mask_end_x, mask_start_y:mask_end_y] = action_marginals[start_x:end_x, start_y:end_y]
    cmap = plt.cm.get_cmap('binary')  # Example: using the 'viridis' colormap

    norm = colors.Normalize(vmin=0, vmax=15)
    ax.imshow(mask, alpha=1,  cmap=cmap, norm=norm, zorder=1)
    ax.text(pose[1], pose[0], str('x'), ha='center', va='center', color='black', fontsize=30, alpha=0.8)
    
    return convert_matplot_to_image(fig)

def get_frame(env, pose ):
    fig = plt.figure(1)
    ax = plot_map(env.rooms, cmap=env.rooms_colour_map, show = False)
    ax.text(pose[1], pose[0], str('x'), ha='center', va='center', color='black', fontsize=30)

    return convert_matplot_to_image(fig)

def convert_matplot_to_image(fig):
    fig.canvas.draw()
    # Convert figure to RGB byte string
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig) 
    image = imageio.core.util.Image(image)
    # Save axes to BytesIO buffer
    # buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)

    # # Read image data from BytesIO buffer
    # image = imageio.imread(buffer)
    # textvar.set_visible(False)
    return image

def plot_map(rooms, cmap, show = True, alpha=1.0):
    fig = plt.figure(1)
    ax = plt.subplot(1,1,1)
    ax.imshow(rooms, cmap=cmap, alpha=alpha, zorder=2)
    # Set ticks to show only integer values
    ax.set_xticks(np.arange(rooms.shape[1]))
    ax.set_yticks(np.arange(rooms.shape[0]))
    if show:
        # Annotate the numbers on top of the cells
        for i in range(rooms.shape[0]):
            for j in range(rooms.shape[1]):
                ax.text(j, i, str(rooms[i, j]), ha='center', va='center', color='black', fontsize=30)    
        try:
            plt.savefig('figures/room_'+str(rooms.shape[0])+'x'+str(rooms.shape[1])+'_'+str(np.max(rooms))+'obs.jpg')
        except FileNotFoundError:
            print('the path to figures is inexistant to save the plot_map')
    return ax

def from_policy_to_pose(env, p, policy, add_rand=False):
    agent_poses = [list(p)]
    observations = [env.get_ob_given_p(p)]
    start_range = (20, 40)
    end_range = (-40, 20)
    length_policy = len(policy)
    for a_idx in range(length_policy):
        o, p = env.hypo_step(int(policy[a_idx]), p)

        if add_rand:
            # Calculate the lerp factor based on the current step
            lerp_factor = a_idx / ( length_policy - 1)
            # Use lerp to get values between the start_range and end_range
            pose_noise_x = np.random.uniform(*start_range) + lerp_factor * (np.random.uniform(*end_range) - np.random.uniform(*start_range))
            pose_noise_y = np.random.uniform(*start_range) + lerp_factor * (np.random.uniform(*end_range) - np.random.uniform(*start_range))
        else:
            pose_noise_x = 0
            pose_noise_y = 0
        pose_wt_noise = [p[0] + pose_noise_x / 100, p[1] + pose_noise_y / 100]
        agent_poses.append(pose_wt_noise)
        observations.append(o)

    # agent_poses = np.vstack(agent_poses)
    # observations = np.vstack(observations)

    return agent_poses, observations

def add_random_to_pose(poses):
    start_range = (10, 40)
    end_range = (-40, -10)
    n_poses = len(poses)
    agent_poses = []
    for idx, pose in enumerate(poses):
        lerp_factor = idx / (n_poses - 1)
        # Use lerp to get values between the start_range and end_range
        pose_noise_x = np.random.uniform(*start_range) + lerp_factor * (np.random.uniform(*end_range) - np.random.uniform(*start_range))
        pose_noise_y = np.random.uniform(*start_range) + lerp_factor * (np.random.uniform(*end_range) - np.random.uniform(*start_range))
        pose_wt_noise = [pose[0] + pose_noise_x / 100, pose[1] + pose_noise_y / 100]
        agent_poses.append(pose_wt_noise)
    return agent_poses


def plot_path_in_map(env, pose, policy=None):
    if policy is None:
        agent_poses = add_random_to_pose(pose)
    else:
        agent_poses, obs = from_policy_to_pose(env, pose, policy, add_rand=True)
    ax = plot_map(env.rooms, env.rooms_colour_map, show=False)
    agent_poses = np.vstack(agent_poses)
    path = mpath.Path(np.column_stack([agent_poses[:,1], agent_poses[:,0]]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('cividis'), linewidth=2, ax=ax)
    # plot_name = 'figures/'+ model_name + '/room_'+str(env.rooms.shape[0])+'x'+str(env.rooms.shape[1])+'_'+str(np.max(env.rooms))+'obs_0'
    # count = 0
    # while os.path.exists(plot_name+'.jpg'):
    #     count+=1
    #     plot_name = plot_name.replace('obs_'+str(count-1), 'obs_'+str(count))
    #try:
    #     plt.savefig(plot_name +'.jpg')
    # except FileNotFoundError:
    #     print('the path to figures is inexistant to save the plot_path_in_map')
    return ax
#==== GRAPH PLOT OURS ====#
def plot_graph(env, A, B, cmap=cm.Spectral, output_file= 'test.pdf', test=True):
    # print('agent.M',agent.M, agent.M.shape)
    Transition_matrix = B[0].sum(2)
    #print(pd.DataFrame(Transition_matrix, index=list(range(0,Transition_matrix.shape[0])), columns=list(range(0,Transition_matrix.shape[0])), dtype=float))
    #Transition_matrix = agent.sum(2)
    if not test:
        Emission_matrix = A[0]
    else:
        Emission_matrix = [onehot(env.rooms[pose[0], pose[1]], len(np.unique(env.rooms))) for pose in env.state_mapping.keys()]
        Emission_matrix = np.array(Emission_matrix).T
    print('A',Emission_matrix, Emission_matrix.shape)
    Transition_matrix /= Transition_matrix.sum(1, keepdims=True)
    #just want to erase the identity matrix
    # np.fill_diagonal(A, 0)
    # print('M', agent.M)
    # print('Transition_matrix:',Transition_matrix, Transition_matrix.shape)
    v = list(range(Transition_matrix.shape[0]))
    
    print('v:',v)
    g = igraph.Graph.Adjacency((Transition_matrix > 0).tolist())
    print('g:',g)
    
    Transition_matrix = np.log10(0.9+Transition_matrix)
    print(Transition_matrix)
    edge_widths = [Transition_matrix[i, j]*30 for i, j in g.get_edgelist()]
    colors = [cmap(np.argmax(Emission_matrix[:,nl]))[:3] for nl in range(Emission_matrix.shape[1])]
    
    # Create the plot with edge widths
    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=30,
        edge_width=edge_widths,  
        margin=50,
    )
    
    return out

def plot_graph_as_cscg(A, agent_state_mapping, cmap,store_path, edge_threshold= 1):
    
    v = [value['state'] for value in agent_state_mapping.values()]
    obs = [value['ob'] for value in agent_state_mapping.values()]
    #print(pd.DataFrame(A, index=list(range(0,A.shape[0])), columns=list(range(0,A.shape[0])), dtype=float))

    g = igraph.Graph.Adjacency((A > 0).tolist())
    # edge_widths = [np.log(A[i, j]+1)*5 for i, j in g.get_edgelist()]
    # edge_widths = [np.log(A[i, j]+1)*5 for i, j in g.get_edgelist()]
    # edge_widths = [x if x>=edge_threshold else 0 for x in edge_widths]
    # print(edge_widths)
    colors = [cmap(nl)[:3] for nl in obs]
    # plot_name = 'figures/'+ specific_str + 'graph_0'
    # count = 0
    # while os.path.exists(plot_name+'.png'):
    #     count+=1
    #     plot_name = plot_name.replace('graph_'+str(count-1), 'graph_'+str(count))
    file = 'connection_graph_edge_Th_'+str(edge_threshold)+'.png'
    out = igraph.plot(
        g,
        store_path / file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=30,
        # edge_width=edge_widths,  
        margin=50,
    )
    return out

#==== GRAPH PLOT CSCG ====#
def plot_cscg_graph(
    A, x, v, n_clones,agent_state_mapping, store_path, cmap=cm.Spectral, vertex_size=30
):
    v_map = [value['state'] for value in agent_state_mapping.values()]
    obs = [value['ob'] for value in agent_state_mapping.values()]
    colors = [cmap(nl)[:3] for nl in obs]
    # if isinstance(x[0], np.ndarray):
        
    if not len(v_map) == len(v) :
        print('states don-t match agent_state_mapping')
        try:

            poses = list(agent_state_mapping.keys())
            obs = []
            for s in v:
                if s in v_map:
                    index = v_map.index(s)
                else:
                    index = bisect.bisect_left(v_map, s) -1
                if index < 0:
                    index =0
                obs.append(agent_state_mapping[poses[index]]['ob'])
            colors = [cmap(nl)[:3] for nl in obs]
            # if isinstance(x[0], np.ndarray):
            #     x = x[:,0]
            # node_labels = np.arange(np.max(x) + 1).repeat(n_clones)[v]
            #colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
        except ValueError:
            print('Value error in plot_cscg_graph')
            node_labels = np.arange(np.min([A.shape[0], len(n_clones)])).repeat(n_clones)[v]        
            colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]

    g = igraph.Graph.Adjacency((A > 0).tolist())
    
    # plot_name = 'figures/'+ specific_str + 'graph_0'
    # count = 0
    # while os.path.exists(plot_name+'.png'):
    #     count+=1
    #     plot_name = plot_name.replace('graph_'+str(count-1), 'graph_'+str(count))

    out = igraph.plot(
        g,
        store_path / 'connection_graph.png',
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )

    return out

#==== BELIEVES PLOTS OURS====#

def plot_beliefs(Qs, title=""):
    #values = Qs.values[:, 0]
    plt.grid(zorder=0)
    plt.bar(range(Qs.shape[0]), Qs, color='r', zorder=3)
    plt.xticks(range(Qs.shape[0]))
    plt.title(title)
    plt.show()

def plot_likelihood(A, state_mapping=None, tittle_add=''):
    state_labels = [i for i in range(A.shape[1])]
    if state_mapping:
        sorted_state_mapping = dict(sorted(state_mapping.items(), key=lambda x: x[1]['state']))
        ob_labels = [next((key, value['state']) for i in range(A.shape[1]) if value['state'] == i) for key, value in sorted_state_mapping.items() ]
    else:
        ob_labels = [i for i in range(A.shape[0])]
    fig = plt.figure()
    ax = sns.heatmap(A, xticklabels =state_labels , yticklabels = ob_labels, cbar = False)
    plt.title(tittle_add + " Likelihood distribution (A)")
    plt.show()
    
def plot_empirical_prior(B, actions_dict):
    fig, axes = plt.subplots(3,2, figsize=(8, 10))
    actions = actions_dict.keys()
    count = 0
    for i in range(3):
        for j in range(2):
            if count >= 5:
                break
                
            g = sns.heatmap(B[:,:,count], cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i,j])

            g.set_title(actions[count])
            count += 1
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()

def plot_transition(B, actions, state_mapping=None, sorted_labels=True):
    if state_mapping is None:
        labels = [i for i in range(B.shape[1])]
    else:
        labels = [(key, value['state']) for key, value in state_mapping.items()]
        if sorted_labels:
            labels = sorted(labels, key=lambda x: (x[0][0], x[0][1]))
    n_actions = len(actions)
    fig, axes = plt.subplots(2,int(np.ceil(n_actions/2)), figsize = (23,12))
    count = 0
    
    for i in range(2):
        for j in range(int(np.ceil(n_actions/2))):
            if count >= n_actions:
                break 
            #print(i,j)
            action_str = next(key for key, value in actions.items() if value == count)
            # Sorting labels and matrix columns based on the sum of each column
            
            # Plotting the heatmap
            g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)


            #g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
            g.set_title(action_str)
            g.set_xlabel('prev state')
            g.set_ylabel('next state')
            count +=1 
    # fig.delaxes(axes.flatten()[2+int(len(a)/2)])
    plt.tight_layout()
    plt.show()
    
def plot_transition_detailed(B, actions, state_map, desired_state_mapping, model_name, plot=True, save=False):
    
    # pose = next(key for key, value in state_map.items() if value['state'] == i)
    labels = [(key, value['state']) for key, value in state_map.items() ]
    #labels = [next((key, value['state']) for key, value in state_map.items() if value['state'] == i) for i in range(B.shape[1])]
    labels = sorted(labels, key=lambda x: (x[0][0], x[0][1]))
      
    n_actions = len(actions)
    fig, axes = plt.subplots(2,int(np.ceil(n_actions/2)), figsize = (15,8))
    count = 0
    
    for i in range(2):
        for j in range(int(np.ceil(n_actions/2))):
            if count >= n_actions:
                break 

            action_str = next(key for key, value in actions.items() if value == count)
            if 'cscg' in model_name:
                temp_b = cscg_T_B_to_ideal_T_B(B, count, desired_state_mapping, state_map)
            else:
                temp_b = T_B_to_ideal_T_B(B, count, desired_state_mapping, state_map)


            # Plotting the heatmap
            g = sns.heatmap(temp_b, cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)


            #g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
            g.set_title(action_str)
            g.set_xlabel('prev state')
            g.set_ylabel('next state')
            count +=1 
    # fig.delaxes(axes.flatten()[2+int(len(a)/2)])
    plt.tight_layout()
    if plot:
        plt.show()
    if save:
        plot_name = 'figures/'+ model_name +'/'+ model_name + '_Transition_full_matrix_0'
        count = 0
        while os.path.exists(plot_name+'.jpg'):
            count+=1
            plot_name = plot_name.replace('matrix_'+str(count-1), 'matrix_'+str(count))
        plt.savefig(plot_name +'.jpg')



def plot_transition_detailed_resized(B, actions, state_map, desired_state_mapping, plot=True):
    
    n_states = len(desired_state_mapping)
    
    # pose = next(key for key, value in state_map.items() if value['state'] == i)
    labels = [(key, value['state']) for key, value in state_map.items() ]
    #labels = [next((key, value['state']) for key, value in state_map.items() if value['state'] == i) for i in range(B.shape[1])]
    labels = sorted(labels, key=lambda x: (x[0][0], x[0][1]))
      
    n_actions = len(actions)
    fig, axes = plt.subplots(2,int(np.ceil(n_actions/2)), figsize = (15,8))
    count = 0
    
    for i in range(2):
        for j in range(int(np.ceil(n_actions/2))):
            if count >= n_actions:
                break 

            action_str = next(key for key, value in actions.items() if value == count)
            # if 'cscg' in model_name:
            #     temp_b = cscg_T_B_to_ideal_T_B(B, count, desired_state_mapping, state_map)
            # else:
            temp_b = T_B_to_ideal_T_B(B, count, desired_state_mapping, state_map)
            
            temp_b = temp_b[:n_states,:n_states]

            # Plotting the heatmap
            g = sns.heatmap(temp_b, cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)


            #g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
            g.set_title(action_str)
            g.set_xlabel('prev state')
            g.set_ylabel('next state')
            count +=1 
    # fig.delaxes(axes.flatten()[2+int(len(a)/2)])
    plt.tight_layout()
    if plot:
        plt.show()
    # if save:
    #     plot_name = 'figures/'+ model_name + '/'+ model_name +'_Transition_matrix_0'
    #     count = 0
    #     while os.path.exists(plot_name+'.jpg'):
    #         count+=1
    #         plot_name = plot_name.replace('matrix_'+str(count-1), 'matrix_'+str(count))
    #     plt.savefig(plot_name +'.jpg')
    return plt

def print_transitions(B, actions, show=True):
    B_actions = {}
    for key, value in actions.items():
        try:
            a_T = B[0][:,:,value].round(3)
        except IndexError:
            a_T = B[:,:,value].round(3)
        B_a = pd.DataFrame(a_T, index=list(range(0,a_T.shape[0])), columns=list(range(0,a_T.shape[1])), dtype=float)
        B_actions[key] = B_a
        if show:
            print('         ',key)
            print('    prev_s   ')
            print(B_a)
    return B_actions

#==== OURS: From B to ideal B for visualisation ===#
def T_B_to_ideal_T_B(B, action, desired_state_mapping,agent_state_mapping):
    """ 
    given B and action and the mapping we want, 
    re-organise generated B to match this desired state mapping
    This is usefull for testing purposes
    """

    desired_state_mapping = { k:v for k, v in desired_state_mapping.items() if v in agent_state_mapping.keys() }
    desired_state_mapping = {i: desired_state_mapping[key] for i, key in enumerate(sorted(desired_state_mapping.keys()), start=0)}
    temp_b = np.zeros_like(B[:,:,action])
    for n in range(B[:,:,action].shape[0]):
        for p in range(B[:,:,action].shape[1]):
            
            try:
                n_s = desired_state_mapping[n]
                p_s = desired_state_mapping[p]
                B_n_s = agent_state_mapping[n_s]['state']
                B_p_s = agent_state_mapping[p_s]['state']
                temp_b[n][p] = B[B_n_s,B_p_s,action]
            except KeyError:
                #print(n_s,p_s,n,p)
                #This means that position has NOT been discovered yet
                continue
                #return B[:,:,action]
            
    return temp_b

def B_to_ideal_B(B,actions, desired_state_mapping, agent_state_mapping=None):
    """ rearrange the full B matrix"""
    reshaped_B = np.array([])
    actions = dict(sorted(actions.items(), key=lambda item: item[1]))
    for a in actions.values():
        B_a = T_B_to_ideal_T_B(B, a, desired_state_mapping, agent_state_mapping)
        B_a = B_a.reshape(B_a.shape[0], B_a.shape[1], 1)
        if reshaped_B.shape[0] == 0:
            reshaped_B = B_a
        else:
            reshaped_B = np.append(reshaped_B, B_a, axis=2)

    return reshaped_B

#==== CSCG: From B to ideal B for visualisation ====#

def cscg_T_B_to_ideal_T_B(B, action, desired_state_mapping,agent_state_mapping):
    """ 
    given B and action and the mapping we want, 
    re-organise generated B to match this desired state mapping
    This is usefull for testing purposes
    """
    
    desired_state_mapping = { k:v for k, v in desired_state_mapping.items() if v in agent_state_mapping.keys() }
    desired_state_mapping = {i: desired_state_mapping[key] for i, key in enumerate(sorted(desired_state_mapping.keys()), start=0)}
    temp_b = np.zeros_like(B[action,:,:])
    # print('action', action, 'B shape', B.shape, temp_b.shape)
    for n in range(B[action,:,:].shape[0]):
        for p in range(B[action,:,:].shape[1]):
            
            try:
                n_s = desired_state_mapping[n]
                p_s = desired_state_mapping[p]
                B_n_s = agent_state_mapping[n_s]['state']
                B_p_s = agent_state_mapping[p_s]['state']
                # print('prev pose', n_s, B_n_s, 'next pose', p_s, B_p_s, B[action,B_n_s,B_p_s])
                temp_b[p][n] = B[action,B_n_s,B_p_s]
                
            except KeyError:
                #print(n_s,p_s,n,p)
                #This means that position has NOT been discovered yet
                continue
                #return B[:,:,action]
            
    return temp_b

def cscg_B_to_ideal_B(B,actions, desired_state_mapping, agent_state_mapping):
    """ rearrange the full B matrix"""
    reshaped_B = np.array([])
    for a in actions.values():
        B_a = T_B_to_ideal_T_B(B, a, desired_state_mapping, agent_state_mapping)
        B_a = B_a.reshape(1,B_a.shape[0], B_a.shape[1])
        if reshaped_B.shape[0] == 0:
            reshaped_B = B_a
        else:
            reshaped_B = np.append(reshaped_B, B_a, axis=0)

    return reshaped_B

#==== PLOT CSCG BELIEVES ====#

def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.0):
    n_clones = chmm.n_clones
    E = np.zeros((n_clones.sum(), len(n_clones)))
    last = 0
    for c in range(len(n_clones)):
        E[last : last + n_clones[c], c] = 1
        last += n_clones[c]
    E += pseudocount_E
    norm = E.sum(1, keepdims=True)
    norm[norm == 0] = 1
    E /= norm
    T = chmm.C + pseudocount
    norm = T.sum(2, keepdims=True)
    norm[norm == 0] = 1
    T /= norm
    T = T.mean(0, keepdims=True)
    log2_lik, mess_fwd = forwardE(
        T.transpose(0, 2, 1), E, chmm.Pi_x, chmm.n_clones, x, x * 0, store_messages=True
    )
    return mess_fwd

def place_field(mess_fwd, rc, clone):
    assert mess_fwd.shape[0] == rc.shape[0] and clone < mess_fwd.shape[1]
    field = np.zeros(rc.max(0) + 1)
    count = np.zeros(rc.max(0) + 1, int)
    for t in range(mess_fwd.shape[0]):
        r, c = rc[t]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    count[count == 0] = 1
    return field / count



def plot_progression_T(progression):
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Train progression')
    # ax.set_xlabel('Train progression')
    ax.plot(progression, color="tab:red")
    ax.grid(True)
    return ax










def entropy(pk: np.typing.ArrayLike,
            qk: Optional[np.typing.ArrayLike] = None,
            base: Optional[float] = None,
            axis: int = 0
            ) -> Union[np.number, np.ndarray]:
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=axis)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=axis)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : array_like
        Defines the (discrete) distribution. Along each axis-slice of ``pk``,
        element ``i`` is the  (possibly unnormalized) probability of event
        ``i``.
    qk : array_like, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis: int, optional
        The axis along which the entropy is calculated. Default is 0.

    Returns
    -------
    S : {float, array_like}
        The calculated entropy.

    Examples
    --------

    >>> from scipy.stats import entropy

    Bernoulli trial with different p.
    The outcome of a fair coin is the most uncertain:

    >>> entropy([1/2, 1/2], base=2)
    1.0

    The outcome of a biased coin is less uncertain:

    >>> entropy([9/10, 1/10], base=2)
    0.46899559358928117

    Relative entropy:

    >>> entropy([1/2, 1/2], qk=[9/10, 1/10])
    0.5108256237659907

    """
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        vec = special.entr(pk)
    else:
        qk = np.asarray(qk)
        pk, qk = np.broadcast_arrays(pk, qk)
        qk = 1.0*qk / np.sum(qk, axis=axis, keepdims=True)
        vec = special.rel_entr(pk, qk)
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S























