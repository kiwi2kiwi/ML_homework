import numpy as np
import rdkit
import torch_geometric
from torch_geometric.datasets import MoleculeNet

# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")
data

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

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
molecule = Chem.MolFromSmiles(data[0]["smiles"])
molecule

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
embedding_size = 64


import gnn_arch_ES
model = gnn_arch_ES.GCN(data=data, embedding_size=embedding_size)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 8
loader = DataLoader(data[:int(data_size * 0.7)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
validation_loader = DataLoader(data[int(data_size * 0.7):int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

test_loader = DataLoader(data[int(data_size * 0.8):],
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train(data):
    # Enumerate over the data
    losses = torch.tensor([])
    for batch in loader:
      # Use GPU
      batch.to(device)
      # Reset gradients
      optimizer.zero_grad()
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
      # Calculating the loss and gradients
      loss = loss_fn(pred, batch.y)
      loss.backward()

      losses = torch.cat((losses, loss.unsqueeze(0)),0)
      # Update using the gradients
      optimizer.step()

    return torch.mean(losses), embedding

print("Starting training...")
losses = []
val_losses_plt = torch.tensor([])
stopping = False
epochs = 1000
early_stopper = gnn_arch_ES.EarlyStopper(patience=500, min_delta=0.1)

for epoch in range(epochs):
    loss, h = train(data)
    losses.append(loss)
    if epoch % 10 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")
    with torch.no_grad():
        val_losses = torch.tensor([])
        for batch in validation_loader:
            # Use GPU
            batch.to(device)
            # Passing the node features and the connection info
            pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
            # Calculating the loss and gradients
            loss = loss_fn(pred, batch.y)
            val_losses = torch.cat((val_losses, loss.unsqueeze(0)), 0)
        if early_stopper.early_stop(torch.mean(val_losses)):
            stopping = True
            print("Early stopping at Epoch: ", epoch)
        val_losses_plt = torch.cat((val_losses_plt, torch.mean(val_losses).unsqueeze(0)), 0)
    if stopping:
        break


# Visualize learning (training loss)
import seaborn as sns
import matplotlib.pyplot as plt
losses_float = [np.sqrt(float(loss.cpu().detach().numpy())) for loss in losses]
loss_indices = [i for i,l in enumerate(losses_float)]
ax1 = sns.lineplot(loss_indices, losses_float, label='Training Loss')

val_losses_float = [np.sqrt(float(val_loss.cpu().detach().numpy())) for val_loss in val_losses_plt]
val_loss_indices = [i for i,l in enumerate(val_losses_float)]
ax2 = sns.lineplot(val_loss_indices, val_losses_float, label='Validation Loss')
ax1.set(xlabel='Epochs', ylabel='RMSE')
plt.legend()
plt.show()

import pandas as pd
# Initialize lists to store results for the entire test set
y_real_list = []
y_pred_list = []
y_reals = torch.tensor([])
y_preds = torch.tensor([])

# Loop through the test loader to get batches
with torch.no_grad():  # Disable gradient calculation
    for test_batch in test_loader:
        test_batch.to(device)  # Move the batch to the same device as the model (GPU/CPU)

        # Forward pass through the model
        pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)

        # Store the real and predicted values for this batch
        y_real_list.extend(test_batch.y.tolist())
        y_pred_list.extend(pred.tolist())
        y_reals = torch.cat((y_reals, test_batch.y), 0)
        y_preds = torch.cat((y_preds, pred), 0)

# Convert the accumulated lists into a pandas DataFrame
df = pd.DataFrame()
df["y_real"] = y_real_list
df["y_pred"] = y_pred_list
rmse = np.sqrt(sum((np.array(y_real_list) - np.array(y_pred_list))**2)/(df["y_real"].shape[0]))
rmse_2 = np.sqrt(loss_fn(y_reals, y_preds))
print("RMSE: ", rmse_2.item())

torch.save(model, "../GNN_models/current.model")

# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row[0])
df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
df


plt = sns.scatterplot(data=df, x="y_real", y="y_pred")
plt.set(xlim=(-7, 2))
plt.set(ylim=(-7, 2))
plt.show()

