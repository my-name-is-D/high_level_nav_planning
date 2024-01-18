import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import igraph
import os
import copy
from matplotlib import cm, colors
import matplotlib.collections as mcoll
import matplotlib.path as mpath

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
def plot_map(rooms, cmap, show = True):
    fig = plt.figure(1)
    ax = plt.subplot(1,1,1)
    ax.imshow(rooms, cmap=cmap)
    # Set ticks to show only integer values
    ax.set_xticks(np.arange(rooms.shape[1]))
    ax.set_yticks(np.arange(rooms.shape[0]))
    if show:
        # Annotate the numbers on top of the cells
        for i in range(rooms.shape[0]):
            for j in range(rooms.shape[1]):
                ax.text(j, i, str(rooms[i, j]), ha='center', va='center', color='black', fontsize=30)    
        plt.savefig('figures/room_'+str(rooms.shape[0])+'x'+str(rooms.shape[1])+'_'+str(np.max(rooms))+'obs.jpg')
    return ax

def from_policy_to_pose(env, p, policy, add_rand=False):
    agent_poses = [list(p)]
    observations = [env.get_ob_given_p(p)]
    start_range = (20, 40)
    end_range = (-40, 20)

    for a_idx in range(len(policy)):
        o, p = env.hypo_step(int(policy[a_idx]), p)

        if add_rand:
            # Calculate the lerp factor based on the current step
            lerp_factor = a_idx / (len(policy) - 1)
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

def plot_path_in_map(env, start_pose, policy, cmap):
    agent_poses, obs = from_policy_to_pose(env, start_pose, policy, add_rand=True)
    ax = plot_map(env.rooms, cmap, show=False)
    agent_poses = np.vstack(agent_poses)
    path = mpath.Path(np.column_stack([agent_poses[:,1], agent_poses[:,0]]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('cividis'), linewidth=2, ax=ax)
    plot_name = 'figures/room_'+str(env.rooms.shape[0])+'x'+str(env.rooms.shape[1])+'_'+str(np.max(env.rooms))+'obs_0'
    count = 0
    while os.path.exists(plot_name+'.jpg'):
        count+=1
        plot_name = plot_name.replace('obs_'+str(count-1), 'obs_'+str(count))
    plt.savefig(plot_name +'.jpg')

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

def plot_graph_as_cscg(B, agent_state_mapping, cmap=cm.Spectral, specific_str='', edge_threshold= 1):
    v = [value['state'] for value in agent_state_mapping.values()]
    obs = [value['ob'] for value in agent_state_mapping.values()]
    Transition_matrix = B[0] 
    T = Transition_matrix[v,:][:,v,:]
    A = T.sum(2).round(1)
    A /= A.sum(1, keepdims=True)
    A[A < edge_threshold] = 0
    #print(pd.DataFrame(A, index=list(range(0,A.shape[0])), columns=list(range(0,A.shape[0])), dtype=float))

    g = igraph.Graph.Adjacency((A > 0).tolist())
    # edge_widths = [np.log(A[i, j]+1)*5 for i, j in g.get_edgelist()]
    # edge_widths = [np.log(A[i, j]+1)*5 for i, j in g.get_edgelist()]
    # edge_widths = [x if x>=edge_threshold else 0 for x in edge_widths]
    # print(edge_widths)
    colors = [cmap(nl)[:3] for nl in obs]
    plot_name = 'figures/'+ specific_str + 'graph_0'
    count = 0
    while os.path.exists(plot_name+'.png'):
        count+=1
        plot_name = plot_name.replace('graph_'+str(count-1), 'graph_'+str(count))

    out = igraph.plot(
        g,
        plot_name+'.png',
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
    chmm, x, a, specific_str, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())
    node_labels = np.arange(x.max() + 1).repeat(chmm.n_clones)[v]
    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    plot_name = 'figures/'+ specific_str + 'graph_0'
    count = 0
    while os.path.exists(plot_name+'.png'):
        count+=1
        plot_name = plot_name.replace('graph_'+str(count-1), 'graph_'+str(count))

    out = igraph.plot(
        g,
        plot_name+'.png',
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
        plot_name = 'figures/'+ model_name + '_Transition_full_matrix_0'
        count = 0
        while os.path.exists(plot_name+'.jpg'):
            count+=1
            plot_name = plot_name.replace('matrix_'+str(count-1), 'matrix_'+str(count))
        plt.savefig(plot_name +'.jpg')
    
def plot_transition_detailed_resized(B, actions, n_states, state_map, desired_state_mapping, model_name, plot=True, save=False):
    
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
    if save:
        plot_name = 'figures/'+ model_name + '_Transition_matrix_0'
        count = 0
        while os.path.exists(plot_name+'.jpg'):
            count+=1
            plot_name = plot_name.replace('matrix_'+str(count-1), 'matrix_'+str(count))
        plt.savefig(plot_name +'.jpg')

def print_transitions(B, actions):
    for key, value in actions.items():
        print('         ',key)
        print('    prev_s   ')
        try:
            a_T = B[0][:,:,value].round(3)
        except IndexError:
            a_T = B[:,:,value].round(3)
        
        print(pd.DataFrame(a_T, index=list(range(0,a_T.shape[0])), columns=list(range(0,a_T.shape[0])), dtype=float))

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
    for a in actions.values():
        B_a = T_B_to_ideal_T_B(B[0], a, desired_state_mapping, agent_state_mapping)
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




































