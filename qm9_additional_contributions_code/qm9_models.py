import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNRegressor(nn.Module):
    """Simple GCN baseline / capacity variant for QM9 regression.

    The model returns a graph-level molecular embedding before the final MLP.
    This is useful for latent-space visualization.
    """

    def __init__(self, in_dim=11, hidden_dim=128, num_layers=3, out_dim=1, dropout=0.0):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data, return_embedding=False):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = global_mean_pool(x, batch)
        out = self.mlp(graph_emb)

        if return_embedding:
            return out, graph_emb
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
