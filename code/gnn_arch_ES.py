import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.model = None
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.model = model
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

from torch_geometric.datasets import MoleculeNet

class GCN(torch.nn.Module):
    def __init__(self, data, embedding_size):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden

# class GATConv(in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1,
# concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, edge_dim: Optional[int] = None,
# fill_value: Union[float, Tensor, str] = 'mean', bias: bool = True, residual: bool = False, **kwargs)

class GAT(torch.nn.Module):
    def __init__(self, data, embedding_size, dropout_rate):
        # Init parent
        super(GAT, self).__init__()
        torch.manual_seed(42)
        self.dropout_rate = dropout_rate

        # GAT layers
        self.gat1 = GATConv(data.num_features, embedding_size*8)
        self.gat2 = GATConv(embedding_size*8, embedding_size*8)
        self.gat3 = GATConv(embedding_size*8, embedding_size*8)
        # self.initial_conv = GCNConv(data.num_features, embedding_size)
        #self.conv1 = GATConv(embedding_size*4, embedding_size*4)
        self.conv2 = GCNConv(embedding_size*8*4, embedding_size*8//4)
        # self.conv3 = GCNConv(embedding_size//2, embedding_size//3)
        # self.conv4 = GCNConv(embedding_size//3, embedding_size//2)

        # Output layer
        self.out = Linear(embedding_size*8//4, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.gat1(x, edge_index)
        hidden = F.tanh(hidden)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)  # Apply dropout


        # Other Conv layers
        hidden = self.gat2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)  # Apply dropout

        hidden = self.gat3(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)  # Apply dropout

        # hidden = self.conv3(hidden, edge_index)
        # hidden = F.tanh(hidden)
        # hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)  # Apply dropout
        # hidden = self.conv4(hidden, edge_index)
        # hidden = F.tanh(hidden)
        # hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)  # Apply dropout

        # Global Pooling (stack different aggregations)
        pooling_hidden = torch.cat([gmp(hidden, batch_index),
                            gmp(-hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        # Repeat pooled features for each node in the graph
        num_nodes_in_batch = batch_index.shape[0]  # Total number of nodes
        num_graphs_in_batch = pooling_hidden.shape[0]  # Number of graphs (batch size)

        # Create a mapping from node index to graph index
        node_to_graph_map = torch.zeros(num_nodes_in_batch, dtype=torch.int64, device=batch_index.device)

        # Iterate and assign graph indices to nodes
        for graph_index in range(num_graphs_in_batch):
            node_to_graph_map[batch_index == graph_index] = graph_index

        # Repeat the pooled features for each node based on the mapping
        repeated_pooling_hidden = pooling_hidden[node_to_graph_map]

        # Concatenate with original node embeddings
        hidden = torch.cat([hidden, repeated_pooling_hidden], dim=1)


        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)  # Apply dropout

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden