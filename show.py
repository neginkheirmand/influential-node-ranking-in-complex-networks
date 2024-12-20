import os
import pickle
import streamlit as st
import matplotlib.pyplot as plt


# Directory where pickles are saved
pickle_dir = "./assets/pickles/"

# Get the list of pickle files
def get_pickle_files(pickle_dir):
    return [
        f for f in os.listdir(pickle_dir) 
        if f.endswith("_visualization.pkl")
    ]


# Display the graphs in Streamlit
st.title("Graph Viewer")
st.header("Select and View Graphs")

# Get the list of available `.pkl` files
pickle_files = get_pickle_files(pickle_dir)

if pickle_files:
    # Remove the "_visualization.pkl" suffix for better readability
    graph_names = [os.path.splitext(f)[0].replace("_visualization", "") for f in pickle_files]

    # Create a dropdown for selecting the graph
    selected_graph = st.selectbox("Select a graph", graph_names)

    # Load and display the selected graph
    if selected_graph:
        st.write(f"Displaying the graph: {selected_graph}")
        pickle_path = os.path.join(pickle_dir, f"{selected_graph}_visualization.pkl")
        if not os.path.exists(pickle_path):
            print(f"Pickle file does not exist: {pickle_path}")
        elif os.path.getsize(pickle_path) == 0:
            print(f"Pickle file is empty: {pickle_path}")
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
