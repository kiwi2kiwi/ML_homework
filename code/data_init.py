from torch_geometric.datasets import MoleculeNet

# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 8
from torch_geometric.data import DataLoader
loader = DataLoader(data[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
validation_loader = DataLoader(data[int(data_size * 0.8):int(data_size * 0.9)],
                    batch_size=10000, shuffle=True)
print(loader.dataset[0])
data = data.shuffle()

loader = DataLoader(data[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
validation_loader = DataLoader(data[int(data_size * 0.8):int(data_size * 0.9)],
                    batch_size=10000, shuffle=True)
print(loader.dataset[0])


print("Dataset type: ", type(data))
print("Dataset features: ", data.num_features)
print("Dataset target: ", data.num_classes)
print("Dataset length: ", data.len)
print("Dataset sample: ", data[0])
print("Sample  nodes: ", data[0].num_nodes)
print("Sample  edges: ", data[0].num_edges)

# Investiagte the features of the node of graph
data[0].x

# Investigating the edges in sparse COO format
# Shape [2, num_edges]
data[0].edge_index.t()

# See the target value of data[0]
data[0].y

data[0]["smiles"]

import torch
torch.save(data, "../data/data_esol.dat")
