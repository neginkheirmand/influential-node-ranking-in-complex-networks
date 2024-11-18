import streamlit as st
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Function to get graph paths
def get_graph_paths(dataset_dir="./datasets/"):
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

# Function to load the graph and compute metrics
def load_graph_and_metrics(graph_file):
    G = nx.read_edgelist(graph_file)
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    max_degree = max(dict(G.degree()).values())
    return G, avg_degree, max_degree

# Streamlit App
st.title("Graph Files Viewer")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Graph Files", "Home" ])

if page == "Home":
    st.header("Welcome to the Home Page")
    st.write("Navigate to the 'Graph Files' page to view the list of graph files.")

elif page == "Graph Files":
    st.header("Graph Files in Dataset Directory")
    dataset_dir = st.text_input("Dataset Directory", "./datasets/")
    
    if st.button("Load Files"):
        graph_files = get_graph_paths(dataset_dir)
        
        if graph_files:
            st.success(f"Found {len(graph_files)} graph files:")
            
            # Convert to DataFrame
            df = pd.DataFrame(graph_files, columns=["Path", "Name"])
            
            # Display the table with clickable links
            st.write("### File Table")
            for path, name in graph_files:
                if st.button(f"View {name}"):
                    # Load graph and compute metrics
                    G, avg_degree, max_degree = load_graph_and_metrics(path)
                    
                    # Display NetworkX Graph visualization
                    st.write(f"### Network Visualization of {name}")
                    
                    # pyvis interactive graph visualization
                    net = Network(notebook=True)
                    net.from_nx(G)
                    net.show("graph.html")
                    st.components.v1.html(open("graph.html", "r").read(), height=600)
                    
                    # Display Graph Metrics in a table
                    st.write(f"### Metrics for {name}")
                    metrics_data = {
                        "Metric": ["Average Degree", "Maximum Degree"],
                        "Value": [avg_degree, max_degree]
                    }
                    metrics_df = pd.DataFrame(metrics_data)
                    st.table(metrics_df)
                    
        else:
            st.warning("No `.edges` files found in the specified directory.")
