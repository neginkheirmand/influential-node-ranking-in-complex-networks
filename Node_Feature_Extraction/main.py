import os
from pathlib import Path
import networkx as nx
import csv
import pandas as pd
from tqdm import tqdm

def file_exists(file_path):
    return os.path.isfile(file_path)

def folder_exists(folder_path):
    return os.path.isdir(folder_path)
  
def get_graph_paths(dataset_dir= './datasets/'):
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


def WiD1(graph_feature_path, G):
    sorted_nodes = sorted(G.nodes())
    # Write the data to the CSV file
    with open(graph_feature_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Node', 'WiD1'])
        
        # Write the node and degree data
        for node in sorted_nodes:
            degree = G.degree(node)
            writer.writerow([node, degree])

def WiD2(graph_feature_path, G):
    df = pd.read_csv(graph_feature_path)
    def compute_WiD2(row):
        node = row['Node']
        neighbors = list(G.neighbors(node))
        
        # Sum the WiD1 value of the node and its neighbors
        total_WiD1 = row['WiD1'] + sum(df[df['Node'].isin(neighbors)]['WiD1'])
        return total_WiD1

    # Apply the function to create the WiD2 column
    df['WiD2'] = df.apply(compute_WiD2, axis=1)
    df.to_csv(graph_feature_path, index=False)

def WiD3(graph_feature_path, G):
    df = pd.read_csv(graph_feature_path)
    def compute_WiD3(row):
        node = row['Node']
        neighbors = list(G.neighbors(node))
        
        # Sum the WiD2 value of the node and its neighbors
        sum_WiD2 = row['WiD2'] + sum(df[df['Node'].isin(neighbors)]['WiD2'])
        return sum_WiD2

    # Apply the function to create the WiD2 column
    df['WiD3'] = df.apply(compute_WiD3, axis=1)

    df.to_csv(graph_feature_path, index=False)

def feature_graph(graph_path, graph_name, feature_path = './datasets/Features/'):
    G = nx.read_edgelist(graph_path, comments="%", nodetype=int)
    graph_feature_path = os.path.join(feature_path, graph_name + '.csv')
    if not file_exists(graph_feature_path):
        WiD1(graph_feature_path, G)
        WiD2(graph_feature_path, G)
        WiD3(graph_feature_path, G)

        print(f"done with {graph_name}, created {graph_feature_path}")


def main():
    feature_path = './datasets/Features/'
    graph_list = get_graph_paths('./datasets/')

    print('graph_list: ')
    for graph in graph_list:
        print(graph)


    for g_path, g_name in tqdm(graph_list, desc="Processing Graphs", unit="graph"):
        feature_graph(g_path, g_name, feature_path)

    print("All graphs have been processed.")


if __name__ == '__main__':
    main()