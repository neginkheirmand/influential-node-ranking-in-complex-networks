import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import numpy as np
import os
import matplotlib.pyplot as plt
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

def file_exists(file_path):
    return os.path.isfile(file_path)

def folder_exists(folder_path):
    return os.path.isdir(folder_path)


def SIR(G, infected, gama=1, num_b=6, num_iterations=100, num_steps=100):
    # Get the mean degree (k) of the graph
    degrees = [deg for _, deg in G.degree()]
    mean_degree = np.mean(degrees)
    # Calculate B_Threshold
    B_Threshold = mean_degree / (mean_degree**2 - mean_degree)
    # Range of B values
    B_values = np.linspace(1 * B_Threshold, 1.9 * B_Threshold, num_b)
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
        affected_scales[round(B, 3)] = round(affected_scale, 3)

        # Plot the trend for each B
        # viz = DiffusionTrend(model, trends[-1])  # Use the last iteration's trends for visualization
        
        # plt.figure()  # Create a new figure for each plot
        # viz.plot()  # Call the plot method of the viz object
        # plt.title(f"Diffusion Trend for B={round(B, 3)}")
        
        # plt.close()  # Close the plot to free memory




# directory = "./datasets/"
# graph_list = []
# for dirpath, _, files in os.walk(directory):
#     for filename in files:
#         try:
#             if filename.endswith(".edges"):
#                 file_path = os.path.join(dirpath, filename) 
#                 graph_list.append(file_path)
#                 graph = nx.read_edgelist(file_path, comments="%", nodetype=int)
#                 b_list = sir_model(graph, 0.1,5,1000)
#                 b_dict = get_Bdict_from_Blist(b_list)
#                 i = 0
#                 for b in b_dict.keys():
#                     b_dict[b]=sorted(b_dict[b], key=lambda x: x[0])
#                     x_  = [t[0] for t in b_dict[b]]
#                     y_  = [t[1] for t in b_dict[b]]
#                     Sir = pd.DataFrame({'Node':x_,'SIR':y_})
#                     Sir.to_csv(f'{filename}.csv',index=False)
#                     i+=1
#         except Exception as e: 
#             print(e, f'{filename}')