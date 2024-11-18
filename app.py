import os
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


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



def load_graph(graph_file):
    G = nx.read_edgelist(graph_file)
    return G

# Function to load the graph and compute metrics
def load_graph_and_metrics(graph_file):
    G = nx.read_edgelist(graph_file)
    avg_degree = sum(dict(G.degree()).values()) / len(G)
    max_degree = max(dict(G.degree()).values())
    return G, avg_degree, max_degree

# Title and Header
st.title("LCNN Graph Management App")
st.header("Welcome to LCNN Graph Management Application")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Graph Files", "Home", "About", "Graph Viewer" ])

# Main Page Logic
if page == "Home":
    st.subheader("Home Page")
    st.write("This is the home page. Here you can interact with the main content.")
    
    # Input Fields
    name = st.text_input("Enter your name", "")
    age = st.number_input("Enter your age", min_value=0, max_value=120, value=25)
    
    if st.button("Submit"):
        st.success(f"Hello, {name}! You are {age} years old.")
        
elif page == "About":
    st.subheader("About Page")
    st.write("This page contains information about the application.")

elif page == "Graph Files":
    st.header("Graph Files in Dataset Directory")
    dataset_dir = st.text_input("Dataset Directory", "./datasets/")
    
    if st.button("Load Files"):
        graph_files = get_graph_paths(dataset_dir)
        
        if graph_files:
            st.success(f"Found {len(graph_files)} graph files:")
            
            # Convert to DataFrame
            df = pd.DataFrame(graph_files, columns=["Path", "Name"])
            
            # Add clickable links to the table
            df["Card Link"] = df["Name"].apply(
                lambda name: f'<a href="#{name}" style="text-decoration: none; color: #1e90ff;">{name}</a>'
            )
            st.write("### File Table")
            st.write(
                df[["Card Link", "Path"]].rename(columns={"Card Link": "Name"}).to_html(escape=False, index=False),
                unsafe_allow_html=True,
            )
            
            # Add cards below the table with better styling
            st.write("### File Cards")
            for path, name in graph_files:
                st.markdown(f"""
                <div id="{name}" style="padding: 15px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f1f1f1; color: #333;">
                    <h4 style="color: #333;">{name}</h4>
                    <p><strong>Path:</strong> {path}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No `.edges` files found in the specified directory.")
elif page == "Graph Viewer":
    st.header("Graph Viewer")

    # Get the list of available graph files
    dataset_dir = "./datasets/"
    graph_files = get_graph_paths(dataset_dir)

    if not graph_files:
        st.warning("No graph files found in the dataset directory.")
    else:
        # Create a dropdown menu for selecting a graph file
        graph_names = [name for _, name in graph_files]
        selected_graph = st.selectbox("Select a graph to view:", graph_names)

        # Add a button to trigger graph loading
        if st.button("Load Graph"):
            # Get the full path of the selected graph
            selected_file_path = next(
                (path for path, name in graph_files if name == selected_graph), None
            )

            if selected_file_path:
                try:
                    # Load and display the selected graph
                    G = nx.read_edgelist(selected_file_path)

                    # Calculate node positions
                    pos = nx.spring_layout(G)

                    # Create a color map for the nodes
                    color_map = ['skyblue' for _ in G.nodes()]

                    # Plot the graph
                    st.write(f"### Graph Visualization: {selected_graph}")
                    plt.figure(figsize=(10, 7))
                    nx.draw(
                        G, pos, node_color=color_map, with_labels=True, edge_color='gray', font_size=10
                    )
                    st.pyplot(plt)

                except Exception as e:
                    st.error(f"Failed to load or plot the graph: {e}")
