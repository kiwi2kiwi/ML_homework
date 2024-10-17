import numpy as np
import rdkit
import torch_geometric
import torch
data = torch.load("../data/data_esol.dat")
import random
def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(0)
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
model = gnn_arch_ES.GAT(data=data, embedding_size=embedding_size, dropout_rate=0.2)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00015)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 8
loader = DataLoader(data[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
validation_loader = DataLoader(data[int(data_size * 0.8):int(data_size * 0.9)],
                    batch_size=10000, shuffle=True)

test_loader = DataLoader(data[int(data_size * 0.9):],
                         batch_size=10000, shuffle=True)


def train(data):
    # Enumerate over the data
    losses = torch.tensor([]).to(device)
    for batch in loader:
      # Use GPU
      batch.to(device)
      # Reset gradients
      optimizer.zero_grad()
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)

      #*** Calculate the loss using graph-level targets ***
      # Get the graph-level predictions by taking the mean of node predictions for each graph
      graph_level_pred = global_mean_pool(pred, batch.batch)

      # Calculating the loss and gradients
      loss = loss_fn(graph_level_pred, batch.y)
      loss.backward()

      losses = torch.cat((losses, loss.unsqueeze(0)),0)
      # Update using the gradients
      optimizer.step()
    return torch.mean(losses), embedding



print("Starting training...")
losses = []
val_losses_plt = torch.tensor([]).to(device)
stopping = False
epochs = 1000
early_stopper = gnn_arch_ES.EarlyStopper(patience=500, min_delta=0.05)
# %load_ext memory_profiler
# %memit
for epoch in range(epochs):
    loss, h = train(data)
    losses.append(loss)
    with torch.no_grad():
        val_losses = torch.tensor([]).to(device)
        for batch in validation_loader:
            # Use GPU
            batch.to(device)
            # Passing the node features and the connection info
            pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)

            graph_level_pred = global_mean_pool(pred, batch.batch)

            # Calculating the loss and gradients
            val_loss = loss_fn(graph_level_pred, batch.y)
            val_losses = torch.cat((val_losses, val_loss.unsqueeze(0)), 0)
        if early_stopper.early_stop(torch.mean(val_losses).cpu().detach().numpy(), model):
            stopping = True
            print("Early stopping at Epoch: ", epoch)
        val_losses_plt = torch.cat((val_losses_plt, torch.mean(val_losses).unsqueeze(0)), 0)
    # scheduler.step(torch.mean(val_losses))

    if epoch % 10 == 0:
        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
        #                     locals().items())), key= lambda x: -x[1])[:1]:print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        torch.cuda.empty_cache()
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*0.93
        print("Learning rate: ", g['lr'])
        #print("scheduler last lr: ", scheduler.get_last_lr())
        rl = "%.3f" % np.round(np.sqrt(loss.cpu().detach().numpy()),3)
        vl = "%.3f" % np.round(np.sqrt(val_losses.cpu().detach().numpy()),3)
        print(f"Epoch {epoch} | Train RLoss {rl} | Val RLoss {vl}")
    if stopping:
        break


torch.save(test_loader, "../data/test_loader.ldr")

torch.save(early_stopper.model, "../GNN_models/current.model")

#.cpu().detach().numpy()



# Visualize learning (training loss)
import seaborn as sns
import matplotlib.pyplot as plt
losses_float = [np.sqrt(float(loss.cpu().detach().numpy())) for loss in losses]
loss_indices = [i for i,l in enumerate(losses_float)]
ax1 = sns.lineplot(losses_float, label='Training Loss')

val_losses_float = [np.sqrt(float(val_loss.cpu().detach().numpy())) for val_loss in val_losses_plt]
val_loss_indices = [i for i,l in enumerate(val_losses_float)]
ax2 = sns.lineplot(val_losses_float, label='Validation Loss')
ax1.set(xlabel='Epochs', ylabel='RMSE')
plt.legend()
plt.show()

import pandas as pd
# Initialize lists to store results for the entire test set
y_real_list = []
y_pred_list = []
y_reals = torch.tensor([]).to(device)
y_preds = torch.tensor([]).to(device)

import copy
model_end = copy.deepcopy(model)
# Get the best performing model
model = early_stopper.model
# Loop through the test loader to get batches
with torch.no_grad():  # Disable gradient calculation
    for test_batch in test_loader:
        test_batch.to(device)  # Move the batch to the same device as the model (GPU/CPU)

        # Forward pass through the model
        pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
        pred = global_mean_pool(pred, test_batch.batch)
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
# rmse_2 = np.sqrt(loss_fn(y_reals.to(device), y_preds.to(device)))
print("RMSE: ", rmse.item())

#torch.save(model, "models/low06.model")

# Analyze the results for one batch
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    pred = global_mean_pool(pred, test_batch.batch)
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



# You can now analyze the edge_errors dictionary to find edges with high errors.

# Example: Print the average edge error
# average_edge_error = np.mean(list(edge_errors.values()))
# print("Average edge error:", average_edge_error)

# Example: Find the edges with the highest errors
# sorted_edge_errors = sorted(edge_errors.items(), key=lambda item: item[1], reverse=True)
# print("Edges with highest errors:", sorted_edge_errors[:10])

# You can use these edge errors to highlight problematic edges in your visualizations or analyses.
