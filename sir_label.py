import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import numpy as np
import os
import csv
from pathlib import Path
import multiprocessing
import signal
import pandas as pd

# import matplotlib.pyplot as plt
# from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

def file_exists(file_path):
    return os.path.isfile(file_path)

def folder_exists(folder_path):
    return os.path.isdir(folder_path)
  
def save_tuple(tpl, path):
    if not file_exists(path):
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Node', 'SIR'])
            writer.writerow(tpl)

    else: 
        with open(path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(tpl)


def create_SIR_dir(graph_name, result_path = './datasets/SIR_Results/'):
    sir_dir =os.path.join(result_path, graph_name)
    if not folder_exists(sir_dir):
        print("creating: ", sir_dir)
        sir_dir = Path(sir_dir)
        sir_dir.mkdir(parents=True)


def get_graph_paths(dataset_dir= "./datasets/"):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                if not filename.startswith("ba_edgelist") and filename.endswith(".edges"):
                    file_path = os.path.join(dirpath, filename) 
                    graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list

def get_B_Value(G, num_b=3):
    # Get the mean degree (k) of the graph
    degrees = [deg for _, deg in G.degree()]
    mean_degree = np.mean(degrees)
    # Calculate B_Threshold
    B_Threshold = mean_degree / (mean_degree**2 - mean_degree)
    # Range of B values
    B_values = np.linspace(1 * B_Threshold, 1.9 * B_Threshold, num_b)
    # Use numpy's round function
    B_values = np.round(B_values, 3)
    B_values = B_values.tolist()
    return B_values

def SIR(G, infected, B_values, gama=1, num_iterations=100, num_steps=100):
    num_nodes = G.number_of_nodes()
    affected_scales = {}

    for B in B_values:
        recovered_sum = 0  # To store the sum of recovered nodes across all iterations
        
        # Store trends for plotting
        trends = []

        for i in range(num_iterations):
            # Initialize the SIR model
            model = ep.SIRModel(G)
            
            # Configuration setup
            config = mc.Configuration()
            config.add_model_parameter('beta', B)  # Set infection rate to current B
            config.add_model_parameter('gamma', gama)  # Recovery probability = 1
            # config.add_model_initial_configuration("Infected",  {0: 1})  # Start with node 0 infected
            config.add_model_initial_configuration("Infected",  infected)  
            
            # Set the model configuration
            model.set_initial_status(config)
            
            # Run the model until all nodes are either recovered or susceptible
            iteration = model.iteration_bunch(num_steps)
            
            # Store trends for plotting (useful for later visualization)
            trends.append(model.build_trends(iteration))

            # Get the final state after the infection spread
            final_state = iteration[-1]['node_count']
            recovered_nodes = final_state[2]  # Index 2 represents 'Recovered' nodes
            
            recovered_sum += recovered_nodes
        
        # Calculate the affected scale for the current B
        affected_scale = recovered_sum / (num_iterations * num_nodes)
        affected_scales[B] = round(affected_scale, 3)

        # Plot the trend for each B
        # viz = DiffusionTrend(model, trends[-1])  # Use the last iteration's trends for visualization
        
        # plt.figure()  # Create a new figure for each plot
        # viz.plot()  # Call the plot method of the viz object
        # plt.title(f"Diffusion Trend for B={round(B, 3)}")
        
        # plt.close()  # Close the plot to free memory
    return affected_scales



            # def get_sir_dict(sir_of_graph, affected_scales, node):
            #     b_list = affected_scales.keys()
            #     for b in b_list:
            #         sir_of_graph[b].append((node, affected_scales[b]))
            #     return sir_of_graph

def get_sir_graph_paths(graph_path, num_b=3,  result_path = './datasets/SIR_Results/'):
    graph_name = os.path.splitext(os.path.basename(graph_path))[0]
    paths= []
    for i in range(num_b):
        sir_dir =os.path.join(result_path, graph_name)
        sir_dir = os.path.join(sir_dir, f'{i}.csv')
        paths.append(sir_dir)
    return paths

def get_previously_sim_values(sir_graph_path):
    df = pd.read_csv(sir_graph_path)
    values = df['Node'].tolist()
    return values


#TODO CHECK WHETHER IT ALREADY EXISTS
def add_tuples(node, paths,  affected_scales, result_path = './datasets/SIR_Results/'):
    i = 0
    for b in affected_scales.keys():
        save_tuple((node, affected_scales[b]), paths[i] )
        i+=1

def Sir_of_graph(graph_path, num_b = 3, result_path = './datasets/SIR_Results/'):
    G = nx.read_edgelist(graph_path, comments="%", nodetype=int)
    B_values =get_B_Value(G, num_b)
    paths = get_sir_graph_paths(graph_path, num_b, result_path)

    nodes = sorted(G.nodes())
    if file_exists(paths[0]):
        prev_nodes = get_previously_sim_values(paths[0])
        nodes = list(set(nodes) - set(prev_nodes))
        print("Continuing from where we left off graph: ", graph_path, 'node: ', nodes[0])

    for node in nodes:
        # process node
        infected = {node: 1}
        affected_scales = SIR(G, infected, B_values)
        add_tuples(node, paths, affected_scales, result_path)
        print('added node ', node, 'from ', graph_path)


# Sir_of_graph('./datasets/BA_EXP/ba_edgelist_exp3_4000_10.edges')

# G = nx.read_edgelist('./datasets/BA_EXP/ba_edgelist_exp3_4000_10.edges', comments="%", nodetype=int)
# B_values =get_B_Value(G)
# print(B_values)
# affected_scales = SIR(G, {1: 1}, B_values)
# print(affected_scales)



def process_graph(args):
    g_path, g_name, result_path = args
    print(g_name)
    Sir_of_graph(g_path, num_b=3, result_path=result_path)

def init_worker():
    # Ignore SIGINT in the child processes to allow graceful termination in the parent
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    graph_list = get_graph_paths()
    result_path = './datasets/SIR_Results/'

    # Preprocessing: create directories for each graph
    for (g_path, g_name) in graph_list:
        create_SIR_dir(g_name, result_path)

    # Set the number of processes to 2
    pool_size = 2

    # Create a pool of workers, using init_worker to handle SIGINT correctly
    with multiprocessing.Pool(processes=pool_size, initializer=init_worker) as pool:
        try:
            # Prepare arguments as tuples (g_path, g_name, result_path) for each graph
            args_list = [(g_path, g_name, result_path) for (g_path, g_name) in graph_list]
            
            # Use pool.map to distribute tasks and run up to 4 processes concurrently
            pool.map(process_graph, args_list)

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected, terminating pool...")
            pool.terminate()  # Terminate all running processes
            pool.join()       # Wait for processes to finish cleanup

        else:
            pool.close()      # Close the pool normally if no errors
            pool.join()       # Wait for processes to finish

    print("All graphs have been processed or terminated.")


if __name__ == '__main__':
    multiprocessing.freeze_support()  
    main()