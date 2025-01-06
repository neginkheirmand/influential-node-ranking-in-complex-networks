import pandas as pd
import networkx as nx
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score
import os
import re
import json



def list_folders_in_path(path):
    """
    Prints all the folders in the given path.

    Args:
        path (str): The directory path to search for folders.
    """
    try:
        # List all directories in the path
        folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        
        # Print each folder
        # print("Folders in path:", path)
        # for folder in folders:
            # print(folder)
        return folders
    except FileNotFoundError:
        print(f"Error: The path '{path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_numbers_from_string(input_string):
    """
    Extracts all numbers from a given string and returns them as a list of integers.

    Args:
        input_string (str): The input string to extract numbers from.

    Returns:
        list: A list of integers extracted from the input string.
    """
    # Find all numbers in the string using a regular expression
    numbers = re.findall(r'\d+', input_string)
    # Convert the numbers to integers
    return list(map(int, numbers))


# Example usage
path = "./../data"  # Replace with your desired path
models = list_folders_in_path(path)
model_params = [extract_numbers_from_string(g) for g in models] 

for i in range(len(models)):
    print(f'{i}) {model_params[i]} : {models[i]}')


    
def get_df_csv_files(path):
    """
    Returns a list of all files in the given path that end with 'df.csv'.

    Args:
        path (str): The directory path to search for files.

    Returns:
        list: A list of filenames ending with 'df.csv'.
    """
    return [file for file in os.listdir(path) if file.endswith('df.csv')]

def get_ba_df_csv_files(path):
    return [file for file in os.listdir(path) if file.endswith('df.csv') and file.startswith('ba')]

def cl(input_str):
    
    # Remove '_df.csv' if it exists
    input_str = input_str.replace('_df.csv', '')

    if input_str.startswith('ba_edgelist_exp'):
        parts = input_str[17:].split('_')
        if len(parts) >= 2:
            return f"BA {parts[0]}_{parts[1]}"
        return input_str  # Fallback if the format is unexpected
    else:
        return input_str.split('.')[0]

folder_paths = []

for i in range(len(models)):
# for i in range(4):
    # print(f'{i}) {model_params[i]} : {models[i]}')
    _model_L = model_params[i][0]
    sir_alpha = model_params[i][2]
    save_folder = f'./../data/test_L{_model_L}_b4_sir{sir_alpha}'
    folder_paths.append(save_folder)


df = get_df_csv_files(folder_paths[0])


model_files = {cl(item):item for item in df}  # Replace with actual file names



def file_exists(file_path):
    return os.path.isfile(file_path)


def get_test_graph_paths(dataset_dir= "./../../datasets/"):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                if filename.endswith(".edges") :
                    if filename.startswith("ba_edgelist_exp") or not filename.startswith("ba_edgelist"):
                        file_path = os.path.join(dirpath, filename) 
                        graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list



def get_graph_path(graph_list, graph_name):
    for graph in graph_list:
        if graph[1]==graph_name:
            return graph[0]
    return None

def get_sir_paths(net_name, sir_alpha=0,  num_b=3,  result_path = './../../datasets/SIR_Results/'):
    paths= []
    for i in range(num_b):
        sir_dir =os.path.join(result_path, net_name)
        sir_dir = os.path.join(sir_dir, f'{i}.csv')
        if file_exists(sir_dir):
            paths.append(sir_dir)
    #todo
    if sir_alpha<3 and sir_alpha>=0:
        return paths[sir_alpha]
    
    return paths[1]

skip_graphs= ['p2p-Gnutella04','CA-HepTh', 'arenas-pgp', 'powergrid','NS', 'faa', 'ChicagoRegional', 'ia-crime-moreno', 'maybe-PROTEINS-full', 'sex']

test_folder = f'test_L{_model_L}_b4_sir{sir_alpha}'


test_graph_list = get_test_graph_paths()
test_graph_list = [item for item in test_graph_list if item[1] not in skip_graphs]
# print("present graphs: ")
# for g in test_graph_list:
#     print(g)


# g_name = test_graph_list[0][1]
# graph_path = get_graph_path(test_graph_list, g_name)
# g_test = nx.read_edgelist(graph_path, comments="%", nodetype=int)
# g_sir_path = get_sir_paths(g_name)
# print(g_name)
# print(graph_path)
# print(g_test)
# print(g_sir_path)


import networkx as nx
import pandas as pd
import json
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score

# Initialize a list to store results
results = []

for i in range(len(test_graph_list)):
    g_name = test_graph_list[i][1]
    graph_path = get_graph_path(test_graph_list, g_name)
    g_test = nx.read_edgelist(graph_path, comments="%", nodetype=int)
    g_sir_path = get_sir_paths(g_name)
    print(g_name)
    print(graph_path)
    print(g_test)
    print(g_sir_path)

    # Step 1: Load the graph
    G = nx.read_edgelist(graph_path)  # Replace with your graph

    # Step 2: Calculate closeness centrality
    closeness = nx.closeness_centrality(G)
    closeness_df = pd.DataFrame(list(closeness.items()), columns=['Node', 'Closeness'])

    # Step 3: Get top 10% and 20% based on closeness
    closeness_df = closeness_df.sort_values(by='Closeness', ascending=False)
    top_10_closeness = closeness_df.head(int(len(closeness_df) * 0.1))
    top_20_closeness = closeness_df.head(int(len(closeness_df) * 0.2))

    # Step 4: Load SIR results and clean unnecessary columns
    sir_df = pd.read_csv(g_sir_path)
    sir_df = sir_df[['Node', 'SIR']]  # Keep only the necessary columns
    # Ensure 'Node' column in both DataFrames has the same data type
    sir_df['Node'] = sir_df['Node'].astype(int)
    closeness_df['Node'] = closeness_df['Node'].astype(int)

    # Step 5: Sort SIR results and extract top 10% and 20%
    top_10_sir = sir_df.head(int(len(sir_df) * 0.1))
    top_20_sir = sir_df.head(int(len(sir_df) * 0.2))

    # Step 6: Updated function to compare rankings
    def compare_rankings(top_closeness, top_sir, common_nodes):
        print(len(common_nodes))
        if not common_nodes:
            print("Warning: No common nodes between the two ranking methods.")
            return 0, 0, 0  # Default values when no overlap exists

        # Filter the common nodes from the datasets
        close_df = top_closeness[top_closeness['Node'].isin(common_nodes)].sort_values('Node')
        sir_df_filtered = top_sir[top_sir['Node'].isin(common_nodes)].sort_values('Node')

        # Get rankings
        close_rank_values = close_df['Closeness'].values
        sir_rank_values = sir_df_filtered['SIR'].values

        # Spearman's rank correlation
        spearman_corr, _ = spearmanr(close_rank_values, sir_rank_values)
        
        # Kendall's τ
        kendall_corr, _ = kendalltau(close_rank_values, sir_rank_values)
        
        # MAP calculation
        relevance = [1 if node in common_nodes else 0 for node in top_sir['Node']]
        scores = closeness_df[closeness_df['Node'].isin(top_sir['Node'])].sort_values('Node')['Closeness'].values
        map_score = average_precision_score(relevance, scores)
        
        return spearman_corr, kendall_corr, map_score

    # Step 6: Get common nodes
    common_nodes = set(closeness_df['Node']).intersection(set(sir_df['Node']))

    # Compute metrics
    metrics_10 = compare_rankings(top_10_closeness, top_10_sir, common_nodes)
    metrics_20 = compare_rankings(top_20_closeness, top_20_sir, common_nodes)

    # Save results for this iteration
    results.append({
        "graph_name": g_name,
        "metrics_top_10": {
            "spearman": metrics_10[0],
            "kendall": metrics_10[1],
            "map": metrics_10[2]
        },
        "metrics_top_20": {
            "spearman": metrics_20[0],
            "kendall": metrics_20[1],
            "map": metrics_20[2]
        }
    })
    
    # Save all results to a JSON file
    with open('closeness_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Display results
    print("For Top 10% Nodes:")
    print(f"Spearman's Rank Correlation: {metrics_10[0]}")
    print(f"Kendall's Tau: {metrics_10[1]}")
    print(f"Mean Average Precision (MAP): {metrics_10[2]}")

    print("For Top 20% Nodes:")
    print(f"Spearman's Rank Correlation: {metrics_20[0]}")
    print(f"Kendall's Tau: {metrics_20[1]}")
    print(f"Mean Average Precision (MAP): {metrics_20[2]}")
    print("*****************************************************************88")

print("Results saved to 'closeness_results.json'.")
