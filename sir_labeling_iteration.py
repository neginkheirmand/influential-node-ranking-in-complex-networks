import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import json

def get_graph_paths(dataset_dir= "./datasets/"):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                if filename.endswith(".edges"):
                    file_path = os.path.join(dirpath, filename) 
                    graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list


def load_graph(file_path):
    """Load a graph from an edge list file."""
    try:
        G = nx.read_edgelist(file_path)
        return G
    except Exception as e:
        print(f"Error loading graph {file_path}: {e}")
        return None

def sort_graph_list_by_nodes(graph_list):
    """Sort graph list based on the number of nodes."""
    graph_list_with_sizes = []
    
    for file_path, name in graph_list:
        G = load_graph(file_path)
        if G is not None:
            num_nodes = G.number_of_nodes()
            graph_list_with_sizes.append((file_path, name, num_nodes))
    
    # Sort by the number of nodes (ascending order)
    graph_list_with_sizes.sort(key=lambda x: x[2])
    
    # Return the sorted list (without the size)
    return [(file_path, name) for file_path, name, _ in graph_list_with_sizes]




def get_sir_graph_paths(net_name, num_b=3,  result_path = './datasets/SIR_Results/'):
    paths= []
    for i in range(num_b):
        sir_dir =os.path.join(result_path, net_name)
        sir_dir = os.path.join(sir_dir, f'{i}.csv')
        paths.append(sir_dir)
    return paths


def SIR(G, infected, B_values, gama=1.0, num_iterations=100, num_steps=200):
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


def analyze_sir_vs_iterations(net_name, G, infected_nodes, B_values, gama, num_iterations_list, num_steps=200):
    execution_times = {str(num_iterations): [] for num_iterations in num_iterations_list}

    plt.figure(figsize=(10, 6))

    for node in infected_nodes:
        infected = {node: 1}
        average_sir_values = []

        for num_iterations in num_iterations_list:
            # Measure the start time
            start_time = time.time()

            # Run the SIR simulation
            affected_scales, _ = SIR(G, infected, B_values, gama, num_iterations, num_steps)
            
            # Measure the end time
            end_time = time.time()
            duration = end_time - start_time  # Duration in seconds

            # Save the execution time
            execution_times[str(num_iterations)].append(duration)

            average_sir_value = sum(affected_scales.values()) / len(B_values)
            average_sir_values.append(average_sir_value)

        plt.plot(num_iterations_list, average_sir_values, marker='o', label=f"Node {node}")

    plt.title(f"IS per node over Iterations: {net_name}")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average SIR Value(Influential Scale)")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(f"./sir_labeling/images/iterations/sir_val_over_iter_{net_name}.png", dpi=300, bbox_inches='tight')

    # plt.show()

    # Calculate average execution times for each num_iterations and return
    avg_execution_times = {num_iterations: round(np.mean(times), 2) for num_iterations, times in execution_times.items()}
    return avg_execution_times



def choose_random_node(G):
    return random.choice(list(G.nodes))


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


skip_graphs= ['p2p-Gnutella04','CA-HepTh', 'arenas-pgp', 'powergrid','NS', 'faa', 'ChicagoRegional', 'ia-crime-moreno', 'maybe-PROTEINS-full', 'sex']

# File path for execution times
execution_times_file = "all_execution_times.json"

# Load execution times if the file exists
if os.path.exists(execution_times_file):
    with open(execution_times_file, "r") as f:
        all_execution_times = json.load(f)
else:
    all_execution_times = {}

graph_list = get_graph_paths()

# Sort the graph list by the number of nodes
graph_list = sort_graph_list_by_nodes(graph_list)
for g in graph_list:
    print(g)

# Example usage
for graph in graph_list:
    G_path = graph[0]
    net_name = graph[1]

    # Skip graphs that are already in all_execution_times
    if net_name in all_execution_times:
        print(f"Skipping {net_name}, already processed.")
        continue

    # Skip graphs that are in skip_graphs
    if net_name in skip_graphs:
        print(f"Skipping {net_name}, between skipable graphs.")
        continue


    print(f"Processing {net_name}...")
    
    G = nx.read_edgelist(G_path, comments="%", nodetype=int)
    
    # Choose 8 random nodes
    infected_nodes = [choose_random_node(G) for _ in range(8)]
    print("Selected nodes:", infected_nodes)

    B_values = get_B_Value(G)
    gama = 1.0
    num_iterations_list = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Analyze and get average execution times
    avg_execution_times = analyze_sir_vs_iterations(net_name, G, infected_nodes, B_values, gama, num_iterations_list)

    # Add to the overall execution times
    all_execution_times[net_name] = avg_execution_times

    # Save execution times to JSON after each graph is processed
    with open(execution_times_file, "w") as json_file:
        json.dump(all_execution_times, json_file, indent=4)
        print(f"Execution times for {net_name} saved to {execution_times_file}")
