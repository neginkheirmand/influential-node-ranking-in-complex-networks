import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Load the visualizations from the pickle file
pickle_filename = "./graph_visualizations.pkl"
visualizations = {}

try:
    with open(pickle_filename, "rb") as f:
        visualizations = pickle.load(f)
    st.success(f"Loaded {len(visualizations)} visualizations from {pickle_filename}")
except Exception as e:
    st.error(f"Failed to load visualizations: {e}")

# Add a page for selecting and displaying a graph
st.title("Graph Viewer")
st.header("Select a Graph to View")

# Create a selection box to choose a graph
graph_names = list(visualizations.keys())
selected_graph = st.selectbox("Select a graph", graph_names)

# Plot the selected graph
if selected_graph:
    st.write(f"Displaying the graph: {selected_graph}")
    
    # Retrieve the figure from the visualizations dictionary
    fig = visualizations.get(selected_graph)

    if fig:
        # Display the figure in the Streamlit app
        st.pyplot(fig)
    else:
        st.error("Graph visualization not found.")
