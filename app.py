import os
import pickle
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
                if filename.endswith(".edges") and (filename.startswith("ba_edgelist_exp") or not filename.startswith("ba_edgelist")  ):
                    file_path = os.path.join(dirpath, filename) 
                    graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list


# Get the list of pickle files
def get_pickle_files(pickle_dir):
    return [
        f for f in os.listdir(pickle_dir) 
        if f.endswith("_visualization.pkl")
    ]

def get_graph_features(graph_name, file_path="graph_info.xlsx"):
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)
        
        # Ensure the graph name is in the DataFrame
        if "graph G" not in df.columns:
            raise KeyError("'Graph Name' column not found in the Excel file.")
        
        # Filter the DataFrame for the requested graph
        graph_info = df[df["graph G"] == graph_name]
        
        if graph_info.empty:
            return None  # Graph not found
        
        # Convert the row to a dictionary of features
        features = graph_info.iloc[0].to_dict()  # Take the first match (if multiple exist)
        return features
    
    except Exception as e:
        print(f"Error reading or processing the file: {e}")
        return None



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
    pickle_dir = "./assets/pickles/"
    # Display the graphs in Streamlit
    st.title("Graph Viewer")
    st.header("Select and View Graphs")

    # Get the list of available `.pkl` files
    pickle_files = get_pickle_files(pickle_dir)

    if pickle_files:
        # Remove the "_visualization.pkl" suffix for better readability
        # graph_names = [os.path.splitext(f)[0].replace("_visualization", "") for f in pickle_files]
        
        graph_names = get_graph_paths("./datasets/")
        for i in range(len(graph_names)):
            graph_names[i]=graph_names[i][1]
        # Create a dropdown for selecting the graph
        selected_graph = st.selectbox("Select a graph", graph_names)

        # Load and display the selected graph
        if selected_graph:
            st.write(f"Displaying the graph: {selected_graph}")
            pickle_path = os.path.join(pickle_dir, f"{selected_graph}_visualization.pkl")
            features  = get_graph_features(selected_graph)

            #add the table 
            if features:
                # Convert features dictionary to a DataFrame for better display
                features_df = pd.DataFrame(features.items(), columns=["Feature", "Value"])
                st.write("### Graph Features")
                st.table(features_df)  # Display the features as a table
            else:
                st.warning(f"No features found for the graph: {selected_graph}")
            try:
                with open(pickle_path, "rb") as f:
                    fig = pickle.load(f)
                st.pyplot(fig)
                
            except EOFError:
                st.error(f"Pickle file is corrupted or incomplete: {pickle_path}")
            except Exception as e:
                st.error(f"An error occurred while loading the pickle: {e}")
    else:
        st.warning("No saved graph visualizations found in the pickle directory.")

