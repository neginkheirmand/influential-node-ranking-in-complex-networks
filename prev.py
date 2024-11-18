import pickle
import os
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from tqdm import tqdm


def get_graph_paths(dataset_dir):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            print(filename)
            try:
                # if not filename.startswith("ba_edgelist") and filename.endswith(".edges"):
                if filename.endswith(".edges"):
                    file_path = os.path.join(dirpath, filename) 
                    graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list

dataset_dir = "./datasets/"
graph_list = get_graph_paths(dataset_dir)
print(graph_list)

# pre-generate graph visualizations
visualizations = {}
for path, name in tqdm(graph_list, desc="Processing Graphs", unit="graph"):
    print(path)
    try:
        # Load the graph
        G = nx.read_edgelist(path)

        # Calculate node positions
        pos = nx.spring_layout(G)

        # Create a color map for the nodes
        color_map = ['skyblue' for _ in G.nodes()]

        # Generate the visualization
        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw(
            G, pos, node_color=color_map, with_labels=True, edge_color='gray', font_size=10, ax=ax
        )

        # Save the figure in a dictionary
        visualizations[name] = fig
        plt.close(fig)  # Close the plot to free up memory
    except Exception as e:
        print(f"Failed to generate visualization for {name}: {e}")

# Save visualizations to a file for future use
with open("graph_visualizations.pkl", "wb") as f:
    pickle.dump(visualizations, f)
