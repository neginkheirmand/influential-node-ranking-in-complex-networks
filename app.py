import os
import pickle
import pandas as pd
import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go


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

def read_sir_csv(filename):
    """
    Reads the SIR results from a CSV file and returns Node and SIR values.
    """
    data = pd.read_csv(filename)
    x = data['Node']  # Node indices
    y = data['SIR']   # SIR values
    return x, y

def get_sir_graph_paths(net_name, num_b=3,  result_path = './datasets/SIR_Results/'):
    paths= []
    for i in range(num_b):
        sir_dir =os.path.join(result_path, net_name)
        sir_dir = os.path.join(sir_dir, f'{i}.csv')
        paths.append(sir_dir)
    return paths


# Function to plot node importance using SIR for different B values
def plot_interactive_sir(graph_path, sir_csv_paths):
    fig = go.Figure()
    
    G = nx.read_edgelist(graph_path, comments="%", nodetype=int)
    num_nodes = G.number_of_nodes()
    b_list = get_B_Value(G, len(sir_csv_paths))


    colors = ['red', 'green', 'blue', 'yellow', 'black', 'cyan', 'magenta']
    
    for i, sir_csv in enumerate(sir_csv_paths):
        x, y = read_sir_csv(sir_csv)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"B = {b_list[i]}", line=dict(color=colors[i % len(colors)])))

    fig.update_layout(
        title="Node Importance using SIR for Different B Values",
        xaxis_title="Node Index",
        yaxis_title="Influential Scale (IS)",
        legend_title="B Values",
        height=600,
        width=1200,
    )
    return fig


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

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "About","Graph Files", "Graph Viewer", "SIR analyzer"])

# Main Page Logic
if page == "Home":
    st.subheader("Spreading Influence Identification Home Page")
    st.write("""Welcome to the LCNN Graph Management Application! """)    
        
elif page == "About":
    st.subheader("About Page")
    st.write("""Welcome to the LCNN Graph Management Application! """)
    st.write("""This tool is designed to empower researchers, analysts, and enthusiasts working with complex networks. With intuitive navigation and advanced visualization features, you can:""")
    st.write("""- Explore graph structures and analyze their properties.""")
    st.write("""- Visualize and interpret node importance using the SIR model.""")
    st.write("""- Manage datasets and effortlessly switch between multiple graph files.""")
    st.write("""Whether you're studying information flow, modeling network dynamics, or simply exploring graph theory concepts, this application provides the tools to make your work seamless and insightful.""")
    st.write("""Feel free to reach out with feedback or suggestions—we're constantly evolving to meet your needs!""")

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
    pickle_dir = "./assets/prev_go/"
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
                st.plotly_chart(fig)
                
            except EOFError:
                st.error(f"Pickle file is corrupted or incomplete: {pickle_path}")
            except Exception as e:
                st.error(f"An error occurred while loading the pickle: {e}")
    else:
        st.warning("No saved graph visualizations found in the pickle directory.")
elif page=="SIR analyzer":
    st.header("My Graph Influential Scale Analyzer")
    st.title("My Graph Influential Scale Analyzer")
    st.header("Select and View SIR simulation Results of a graph")

    # Get the list of available `.pkl` files
    graph_list = get_graph_paths()
    if graph_list:
        graph_names = [name for _, name in graph_list]
    
        selected_graph_name = st.selectbox("Select a graph", graph_names)

        # Load and display the selected graph
        if selected_graph_name:
            st.write(f"Displaying the graph: {selected_graph_name}")
            selected_graph_path = next(path for path, name in graph_list if name == selected_graph_name)
            G = load_graph(selected_graph_path)
            b_values = get_B_Value(G)
            # Create a DataFrame with B values as a single row
            b_values_df = pd.DataFrame([b_values], columns=[f'B_{i+1}' for i in range(len(b_values))])

            # Display the table in Streamlit
            st.title("B Values Table")
            st.table(b_values_df)
            sir_plot = plot_interactive_sir(selected_graph_path, get_sir_graph_paths(selected_graph_name))
            st.plotly_chart(sir_plot)


