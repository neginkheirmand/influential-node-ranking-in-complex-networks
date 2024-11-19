import pickle
import os
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from tqdm import tqdm


def get_graph_paths(dataset_dir):
    # TODO: These big files werent done 
    nop = ["ia-crime-moreno", "maybe-PROTEINS-full", "sex", "ChicagoRegional"]
    # yup = ["faa","politician_edges","Stelzl","tvshow_edges","vidal"]
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                name = os.path.splitext(filename)[0]
                if filename.endswith(".edges") and not (name in nop) :
                # if filename.endswith(".edges") and not (name in nop) and name in yup:
                    print(name)
                    file_path = os.path.join(dirpath, filename) 
                    graph_list.append((file_path, name))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list


dataset_dir = "./datasets/"
graph_list = get_graph_paths(dataset_dir)

# Create a folder to save individual pickle files if it doesn't exist
pickle_dir = "./assets/pickles/"
os.makedirs(pickle_dir, exist_ok=True)

# Pre-generate graph visualizations
for path, name in tqdm(graph_list, desc="Processing Graphs", unit="graph"):
    pickle_filename = os.path.join(pickle_dir, f"{name}_visualization.pkl")
    
    if os.path.exists(pickle_filename):
        print(f"Skipping {name}, already exists: {pickle_filename}")
        continue  # Skip graphs already in pickle

    print(f"Processing {name}")
    try:
        # Load the graph
        G = nx.read_edgelist(path)

        G.remove_edges_from(nx.selfloop_edges(G))

        # Calculate node positions
        pos = nx.spring_layout(G)

        # Create a color map for the nodes
        color_map = ['skyblue' for _ in G.nodes()]

        # Generate the visualization
        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw(
            G, pos, node_color=color_map, with_labels=True, edge_color='gray', font_size=10, ax=ax
        )

        # Save the figure in a separate pickle file
        with open(pickle_filename, "wb") as f:
            pickle.dump(fig, f)
        # Verify if the file is written correctly
        if os.path.exists(pickle_filename) and os.path.getsize(pickle_filename) > 0:
            print(f"Pickle file saved successfully: {pickle_filename}")
        else:
            print(f"Failed to save pickle file: {pickle_filename}")


        # Save the PNG file for later use
        png_filename = f"./assets/img/previsualizations/{name}_graph.png"
        fig.savefig(png_filename, dpi=300, bbox_inches="tight")
        print(f"Saved PNG for {name} as {png_filename}")
        
        # Close the plot to free up memory
        plt.close(fig)

    except Exception as e:
        print(f"Failed to generate visualization for {name}: {e}")

print("Individual pickle files and PNGs saved successfully.")
