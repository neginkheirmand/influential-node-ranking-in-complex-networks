import pickle
import os
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from tqdm import tqdm

def get_graph_paths(dataset_dir):
    # Exclude specific large files
    nop = ["ia-crime-moreno", "maybe-PROTEINS-full", "sex", "ChicagoRegional"]
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                name = os.path.splitext(filename)[0]
                if filename.endswith(".edges") and not (name in nop):
                    print(name)
                    file_path = os.path.join(dirpath, filename)
                    graph_list.append((file_path, name))
            except Exception as e:
                print(e, f'{filename}')
    return graph_list

# Define dataset and pickle directories
dataset_dir = "./datasets/"
pickle_dir = "./assets/prev_go/"
os.makedirs(pickle_dir, exist_ok=True)

# Get graph list
graph_list = get_graph_paths(dataset_dir)

# Pre-generate interactive graph visualizations
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

        # Extract node and edge coordinates for Plotly
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create edge traces
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node traces
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"Node {node}")  # Add node label

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                line_width=2
            )
        )

        # Create the interactive Plotly figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'Interactive Graph: {name}',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=30),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)
                        ))

        # Save the Plotly figure to a pickle file
        with open(pickle_filename, "wb") as f:
            pickle.dump(fig, f)

        # Verify the pickle file
        if os.path.exists(pickle_filename) and os.path.getsize(pickle_filename) > 0:
            print(f"Pickle file saved successfully: {pickle_filename}")
        else:
            print(f"Failed to save pickle file: {pickle_filename}")

    except Exception as e:
        print(f"Failed to generate visualization for {name}: {e}")

print("Interactive pickle files saved successfully.")
