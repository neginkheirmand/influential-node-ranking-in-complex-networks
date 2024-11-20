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
from dotenv import load_dotenv
import json


def file_exists(file_path):
    return os.path.isfile(file_path)

def folder_exists(folder_path):
    return os.path.isdir(folder_path)
  
def save_tuple(tpl, path):
    if not file_exists(path):
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Node', 'SIR', 'Infected_sum'])
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
                # if not filename.startswith("ba_edgelist") and filename.endswith(".edges"):
                if filename.endswith(".edges"):
                    file_path = os.path.join(dirpath, filename) 
                    graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list

def get_B_Value(G, num_b=3):
    # Get the mean degree (k) of the graph
    degrees = [deg for _, deg in G.degree()]
    
    # First moment (mean degree)
    mean_degree = np.mean(degrees)

    # Second moment (mean of squared degrees)
    mean_degree_squared = np.mean([deg**2 for deg in degrees])

    # Epidemic threshold (B_Threshold)
    B_Threshold = mean_degree / (mean_degree_squared - mean_degree)
    # Range of B values
    B_values = np.linspace(1 * B_Threshold, 2 * B_Threshold, num_b)
    # Use numpy's round function
    B_values = np.round(B_values, 3)
    B_values = B_values.tolist()
    return B_values

def SIR(G, infected, B_values, gama=1.0, num_iterations=1000, num_steps=200):
# def SIR(G, infected, B_values, gama=1.0, num_iterations=1000, num_steps=200):
    num_nodes = G.number_of_nodes()
    affected_scales = {}
    infected_scales = {}
    for B in B_values:
        recovered_sum = 0  # To store the sum of recovered nodes across all iterations
        infected_sum = 0

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
            
            
            iteration = None
            # Run the model until all nodes are either recovered or susceptible
            for step in range(num_steps):  # Maximum 200 steps
                iteration = model.iteration()
                trends.append(model.build_trends([iteration]))
                
                # Check if all nodes are either recovered or susceptible (no infected nodes left)
                if iteration['node_count'][1] == 0:  # Index 1 corresponds to 'Infected'
                    break  # Exit the loop if no infected nodes remain

            # Get the final state after the infection spread
            final_state = iteration['node_count']
            recovered_nodes = final_state[2]  # Index 2 represents 'Recovered' nodes
            recovered_sum += recovered_nodes
            infected_sum += final_state[1]# Index 1 represents 'inffected' nodes
        
        # Calculate the affected scale for the current B
        affected_scale = recovered_sum / (num_iterations * num_nodes)
        affected_scales[B] = round(affected_scale, 6)
        infected_scales[B] = infected_sum 
    return affected_scales, infected_scales

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
def add_tuples(node, paths,  affected_scales, infected_scales):
    i = 0
    for b in affected_scales.keys():
        save_tuple((node, affected_scales[b], infected_scales[b]), paths[i] )
        i+=1


def Sir_of_graph(graph_path, num_b = 3, result_path = './datasets/SIR_Results/'):
    G = nx.read_edgelist(graph_path, comments="%", nodetype=int)
    B_values =get_B_Value(G, num_b)
    paths = get_sir_graph_paths(graph_path, num_b, result_path)

    nodes = sorted(G.nodes()) 
    prev_siz = 0
    size_ = len(nodes)
    if file_exists(paths[0]):
        prev_nodes = get_previously_sim_values(paths[0])
        prev_siz = len(prev_nodes)
        nodes = list(set(nodes) - set(prev_nodes))
        nodes = sorted(nodes)
        # nodes = sorted(nodes, reverse=True)
        print("Continuing from where we left off graph: ", graph_path, 'node: ', nodes[0],'   size:',  prev_siz,  '/', size_)

    for node in nodes:
        # process node
        infected = {node: 1}
        affected_scales, infected_scales = SIR(G, infected, B_values)
        add_tuples(node, paths, affected_scales, infected_scales)
        prev_siz+=1
        print('added node ', node, 'from ', graph_path,'   size:',  prev_siz,  '/', size_)
    print(f"done with {paths[0]}")

def process_graph(args):
    g_path, g_name, result_path = args
    print(g_name)
    Sir_of_graph(g_path, num_b=3, result_path=result_path)  #TODO: later i can run another 2 representations of it and sum it with the already created ones to get 5 num_b
    print("done with ", g_name)

def init_worker():
    # Ignore SIGINT in the child processes to allow graceful termination in the parent
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():

    load_dotenv(".env")
    machine_name = os.getenv("MACHINE_NAME")
    print('machine_name: ', machine_name)
    result_path = os.getenv("RESULT_ADDRESS")
    print('result_path: ', result_path)
    dataset_dir = os.getenv("DATASET_DIR")
    print('dataset_dir: ', dataset_dir)
    pool_sz = int(os.getenv("POOL_SIZE"))
    print('POOL size: ', pool_sz)

    graph_list = get_graph_paths(dataset_dir)

    with open('machine.json') as f:
        d = json.load(f)
        graph_list = [item for item in graph_list if item[1] in d[machine_name]]

    print('graph_list: ')
    for graph in graph_list:
        print(graph)


    # Preprocessing: create directories for each graph
    for (g_path, g_name) in graph_list:
        create_SIR_dir(g_name, result_path)

    # Set the number of processes depending on the machine
        
    # Create a pool of workers, using init_worker to handle SIGINT correctly
    with multiprocessing.Pool(processes=pool_sz, initializer=init_worker) as pool:
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