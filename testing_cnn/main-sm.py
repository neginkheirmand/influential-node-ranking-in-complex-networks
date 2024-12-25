import json
import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

print("done importing!")

def file_exists(file_path):
    return os.path.isfile(file_path)

def get_graph_all_paths(dataset_dir= "./datasets/"):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                if filename.endswith(".edges") :
                    if filename.startswith("ba_edgelist_exp") or not filename.startswith("ba_edgelist"):
                        file_path = os.path.join(dirpath, filename) 
                        graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list


def get_test_graph_paths(dataset_dir= "./../datasets/"):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                if filename.endswith(".edges") :
                    if not filename.startswith("ba_edgelist"):
                        file_path = os.path.join(dirpath, filename) 
                        graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list

def get_train_graph_paths(dataset_dir= "./../datasets/"):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                if filename.endswith(".edges") :
                    if filename.startswith("ba_edgelist_exp"):
                        file_path = os.path.join(dirpath, filename) 
                        graph_list.append((file_path, os.path.splitext(filename)[0]))
            except Exception as e: 
                print(e, f'{filename}')
    return graph_list


def get_graph_path(graph_list, graph_name):
    for graph in graph_list:
        if graph[1]==graph_name:
            return graph[0]
    return None

def get_sir_paths(net_name, num_b=3,  result_path = './datasets/SIR_Results/'):
    paths= []
    for i in range(num_b):
        sir_dir =os.path.join(result_path, net_name)
        sir_dir = os.path.join(sir_dir, f'{i}.csv')
        if file_exists(sir_dir):
            paths.append(sir_dir)
    #todo
    return paths[0]

def get_feature_path(net_name,  result_path = './datasets/Features/'):
    feature_path =os.path.join(result_path, f'{net_name}.csv')
    if file_exists(feature_path):
        return feature_path
    return None
    

def adjancency_mat(G, node, graph_feature_path, L= 9):
    neighbors = list(G.neighbors(node))
    df = pd.read_csv(graph_feature_path)
    # Ensure the DataFrame is indexed by 'Node' to make lookups easier
    df.set_index('Node', inplace=True)
    
    # Sort neighbors by their WiD3 values
    sorted_neighbors = sorted(neighbors, key=lambda x: df.at[x, 'WiD3'], reverse=True)
    sorted_neighbors.insert(0, node) #insert node at position zero of the list 

    ad_matrix = np.zeros((L, L))
    # Fill the adjacency matrix based on connections in G
    for i, node_i in enumerate(sorted_neighbors[:L]):
        for j, node_j in enumerate(sorted_neighbors[:L]):
            if G.has_edge(node_i, node_j):  # Check if there's an edge between node_i and node_j
                ad_matrix[i, j] = 1  # Set 1 if there is an edge

    return ad_matrix

#TODO: check whether the neighbors should be sorted with the same WiXt
def channel_set(L, adj_matrix, G, graph_feature_path, WiXt,  node):  #wiDt= 'WiD3'
    df = pd.read_csv(graph_feature_path)
    # Ensure the DataFrame is indexed by 'Node' to make lookups easier
    df.set_index('Node', inplace=True)

    neighbors = list(G.neighbors(node))
    # Sort neighbors by their WiD3 values

    # TODO: see what changes if you sort by different things, just remember the sorting for adjacency matrix and this function should be the same
    # sorted_neighbors = sorted(neighbors, key=lambda x: df.at[x, WiXt], reverse=True)
    sorted_neighbors = sorted(neighbors, key=lambda x: df.at[x, 'WiD3'], reverse=True)
    sorted_neighbors.insert(0, node) #insert node at position zero of the list 

    deg_chanl_set = np.zeros((L , L)) 
    for l in range(L): 
        for k in range(L):
            if l == k: 
                deg_chanl_set[l, k] = df.at[node, WiXt]  # WiXt+ alk(which is always 0)
            elif k != 0 and l == 0 and adj_matrix[0, k]: # if adj_matrix[0, k] is 0 then this is a zero-padding and k_node doesnt exist
                k_node = sorted_neighbors[k]
                deg_chanl_set[0, k] = adj_matrix[0, k] * df.at[k_node, WiXt] 
            elif l != 0 and k == 0 and adj_matrix[l, 0]!=0 : 
                l_node = sorted_neighbors[l]
                deg_chanl_set[l, 0] = adj_matrix[l, 0] * df.at[l_node, WiXt] 
            else: 
                deg_chanl_set[l, k] = adj_matrix[l, k] 
    return deg_chanl_set


class NodeDataset(Dataset):
    def __init__(self, G, nodes, graph_feature_path, labels, L):
        self.G = G
        self.nodes = nodes   #TODO: CHECK ITS ALIGNED
        self.graph_feature_path = graph_feature_path
        self.labels = labels  # SIR labels aligned with nodes #TODO: CHECK ITS ALIGNED
        self.L = L

    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        node = self.nodes[idx]  #TODO: CHECK ITS ALIGNED
        
        # Generate adjacency matrix and channel sets on the fly
        adj_matrix = adjancency_mat(self.G, node, self.graph_feature_path, L=self.L)

        degree_channel = np.zeros((3, self.L, self.L))  # 3 layers for WiD1, WiD2, WiD3
        degree_channel[0] = channel_set(self.L, adj_matrix, self.G, self.graph_feature_path, 'WiD1', node)
        degree_channel[1] = channel_set(self.L, adj_matrix, self.G, self.graph_feature_path, 'WiD2', node)
        degree_channel[2] = channel_set(self.L, adj_matrix, self.G, self.graph_feature_path, 'WiD3', node)

        # Similarly for H-index channels
        h_index_channel = np.zeros((3, self.L, self.L))  # 3 layers for WiH1, WiH2, WiH3
        h_index_channel[0] = channel_set(self.L, adj_matrix, self.G, self.graph_feature_path, 'WiH1', node)
        h_index_channel[1] = channel_set(self.L, adj_matrix, self.G, self.graph_feature_path, 'WiH2', node)
        h_index_channel[2] = channel_set(self.L, adj_matrix, self.G, self.graph_feature_path, 'WiH3', node) 
              
        label = self.labels[idx]
        
        # Convert to tensors if using PyTorch
        degree_channel = torch.tensor(degree_channel, dtype=torch.float32) 
        h_index_channel = torch.tensor(h_index_channel, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)    #TODO: MAY NEED TO CHANGE TO FLOAT64
        #  Using float64 can also slow down your model's training time. Most hardware, such as GPUs, is optimized for float32 computations. As a result, using float64 may make operations slower because GPUs are less efficient at handling double-precision computations.
        # wont be using float64(double)

        return degree_channel, h_index_channel, label
    

class InfluenceCNN(nn.Module):
    def __init__(self, input_size):
        super(InfluenceCNN, self).__init__()
        
        # Degree-based channel set convolutional branch
        self.degree_conv = nn.Sequential(
            
            nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1),  # (3, 9, 9) -> (16, 10, 10)
            nn.BatchNorm2d(16),  # Add Batch Normalization
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01), #TODO
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, 10, 10) -> (16, 5, 5)
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),  # (16, 5, 5) -> (32, 6, 6)
            nn.BatchNorm2d(32),  # Add Batch Normalization
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01), #TODO
            nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 6, 6) -> (32, 3, 3)
        )

        # H-index-based channel set convolutional branch
        self.h_index_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),  # Add Batch Normalization
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01), #TODO
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Add Batch Normalization
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01), #TODO
            nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling (2,2)
        )

        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size of each channel is (1, 1), we have 32 channels: (batch_size, 32, 1, 1)

        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(32 * 2, 128),  # Adjusted flattened size: 32 from degree + 32 from H-index
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01), #TODO
            nn.Linear(128, 1)  # Single output for regression
        )

    def forward(self, degree_input, h_index_input):
        # Pass through each convolutional branch
        degree_out = self.degree_conv(degree_input)

        h_index_out = self.h_index_conv(h_index_input)


        # Apply Global Average Pooling
        degree_out = self.global_avg_pool(degree_out)  # Shape: (batch_size, 32, 1, 1)
        h_index_out = self.global_avg_pool(h_index_out)  # Shape: (batch_size, 32, 1, 1)

        # Flatten and concatenate
        degree_out = degree_out.view(degree_out.size(0), -1)
        h_index_out = h_index_out.view(h_index_out.size(0), -1)

        combined = torch.cat((degree_out, h_index_out), dim=1)

        # Fully connected layers for prediction
        output = self.fc(combined)
        
        # Apply sigmoid activation to constrain output to [0, 1]  #TODO: checck cause this wasnt part of the model
        output = torch.sigmoid(output) # this didnt work, why though? i dont know   
        
        return output


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Initialize lists for training and validation data
all_train_nodes, all_train_labels = [], []
all_val_nodes, all_val_labels = []

train_graph_list = get_train_graph_paths()
val_graph_list = get_test_graph_paths()

# Process all training graphs
for graph_name in [graph[1] for graph in train_graph_list]:
    print(f"Processing training graph: {graph_name}")

    # Load paths
    graph_path = get_graph_path(train_graph_list, graph_name)
    sir_list = get_sir_paths(graph_name)
    feature_path = get_feature_path(graph_name)

    if not graph_path or not sir_list or not feature_path:
        print(f"Missing data for training graph {graph_name}. Skipping...")
        continue

    # Load graph and labels
    G = nx.read_edgelist(graph_path, comments="%", nodetype=int)
    labels_df = pd.read_csv(sir_list[0])
    sir_labels = labels_df['SIR'].values
    nodes = labels_df['Node'].values

    # Split into training and validation sets
    train_nodes, _, train_labels, _ = train_test_split(nodes, sir_labels, test_size=0.2, random_state=42)
    all_train_nodes.extend(train_nodes)
    all_train_labels.extend(train_labels)

# Process all validation graphs
for graph_name in [graph[1] for graph in val_graph_list]:
    print(f"Processing validation graph: {graph_name}")

    # Load paths
    graph_path = get_graph_path(val_graph_list, graph_name)
    sir_list = get_sir_paths(graph_name)
    feature_path = get_feature_path(graph_name)

    if not graph_path or not sir_list or not feature_path:
        print(f"Missing data for validation graph {graph_name}. Skipping...")
        continue

    # Load graph and labels
    G = nx.read_edgelist(graph_path, comments="%", nodetype=int)
    labels_df = pd.read_csv(sir_list[0])
    sir_labels = labels_df['SIR'].values
    nodes = labels_df['Node'].values

    # Use all nodes for validation
    all_val_nodes.extend(nodes)
    all_val_labels.extend(sir_labels)

# Create datasets
train_dataset = NodeDataset(G, all_train_nodes, feature_path, all_train_labels, L=9)
val_dataset = NodeDataset(G, all_val_nodes, feature_path, all_val_labels, L=9)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Define model
model = InfluenceCNN(input_size=9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.apply(initialize_weights)
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training and validation
num_epochs = 20
train_losses, val_losses, spearman_scores = [], [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for degree_batch, h_index_batch, label_batch in train_loader:
        degree_batch, h_index_batch, label_batch = degree_batch.to(device), h_index_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        output = model(degree_batch, h_index_batch).squeeze()
        loss = criterion(output, label_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * degree_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for degree_batch, h_index_batch, label_batch in val_loader:
            degree_batch, h_index_batch, label_batch = degree_batch.to(device), h_index_batch.to(device), label_batch.to(device)
            output = model(degree_batch, h_index_batch).squeeze()
            loss = criterion(output, label_batch)
            val_loss += loss.item() * degree_batch.size(0)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # Spearman Rank
    spearman_corr = spearmanr(all_preds, all_labels).correlation if np.std(all_preds) > 0 and np.std(all_labels) > 0 else 0
    spearman_scores.append(spearman_corr)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Spearman: {spearman_corr:.4f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), spearman_scores, label="Spearman Rank Correlation")
plt.xlabel("Epochs")
plt.ylabel("Spearman Rank Correlation")
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), "influence_cnn_single_model.pth")
