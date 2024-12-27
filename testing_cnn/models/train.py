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
from scipy.stats import kendalltau
import os

print("done importing!")

if not torch.cuda.is_available():
    print("GPU UNAVAILABLE")
else:
    print("working on GPU ")

def file_exists(file_path):
    return os.path.isfile(file_path)

def get_graph_all_paths(dataset_dir= "./datasets/"):
    graph_list = []
    for dirpath, _, files in os.walk(dataset_dir):
        for filename in files:
            try:
                if filename.endswith(".edges") :
                    if filename.startswith("ba_edgelist_exp") or not filename.startswith("ba_edgelist") or filename.startswith('ba_edgelist_1000_4'):
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
                    if filename.startswith("ba_edgelist_exp") or not filename.startswith("ba_edgelist"):
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

def get_sir_paths(net_name, sir_alpha=0,  num_b=3,  result_path = './datasets/SIR_Results/'):
    paths= []
    for i in range(num_b):
        sir_dir =os.path.join(result_path, net_name)
        sir_dir = os.path.join(sir_dir, f'{i}.csv')
        if file_exists(sir_dir):
            paths.append(sir_dir)
    #todo
    if sir_alpha<3 and sir_alpha>=0:
        return paths[sir_alpha]
    
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


# ################################################# #
skip_graphs= ['p2p-Gnutella04','CA-HepTh', 'arenas-pgp', 'powergrid','NS', 'faa', 'ChicagoRegional', 'ia-crime-moreno', 'maybe-PROTEINS-full', 'sex']


graph_list = get_graph_all_paths()
graph_list = [item for item in graph_list if item[1] not in skip_graphs]

start_time = time.time()

sir_alpha = 1

graph_name ='ba_edgelist_1000_4'
print(f"{graph_name}")

graph_path = get_graph_path(graph_list, graph_name)
print(graph_path)
sir_list = get_sir_paths(graph_name, sir_alpha)
print(sir_list)
feature_path = get_feature_path(graph_name)
print(feature_path)

graph_feature_path = get_feature_path(graph_name)    #'./data/jazz_Features.csv'
graph_sir_path = sir_list   #'./data/0.csv'

G = nx.read_edgelist(graph_path, comments="%", nodetype=int)

# Load CSV file
labels_df = pd.read_csv(graph_sir_path)

# Extract the SIR column as labels
sir_labels = labels_df['SIR'].values  # Convert to NumPy array for easier handling


hist_output_path = f'./testing_cnn/img/{graph_name}_hist.png'
plt.hist(sir_labels, bins=100, color='blue', alpha=0.7)
plt.xlabel("Influential Scale", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.title("Distribution of Influential Scale values", fontsize=16)
plt.savefig(hist_output_path, dpi=300, bbox_inches='tight')
# plt.show()


# Assuming you have all the nodes and labels loaded properly
train_nodes = labels_df['Node'].values
train_labels = labels_df['SIR'].values

print(train_nodes[1],train_labels[1])   #just checking that its correctly split and gives the correct node

# Create Dataset and DataLoader for Training
train_dataset = NodeDataset(G, train_nodes, graph_feature_path, train_labels, L=9)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Define the Model, Loss Function, and Optimizer:


# Define the model
model = InfluenceCNN(input_size=9)  # Adjust input_size according to your data
# Apply weight initialization
model.apply(initialize_weights)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # For regression
learning_rate = 0.0005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs=200

# Lists to store metrics
train_losses = []

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

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Train Loss: {train_loss} "
    )

# Plotting Training and Validation Loss
loss_output_path = f'./testing_cnn/img/{graph_name}_loss.png'
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", marker='o')
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.title("Training and Validation Loss Over Epochs", fontsize=16)
plt.legend(fontsize=16)
plt.grid()
# plt.show()
plt.savefig(loss_output_path, dpi=300, bbox_inches='tight')

end_time = time.time()
duration = end_time - start_time  # Duration in seconds
print(f"Graph: {graph_name} time: {duration}")

#  Save the Model (optional):
torch.save(model.state_dict(), f'EP200_TRAINED_ba_1000_4_cnn_model_sir{sir_alpha}_raw.pth')
