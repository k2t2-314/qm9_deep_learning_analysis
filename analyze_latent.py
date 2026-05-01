import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance
from torch_geometric.nn import global_mean_pool, global_max_pool
from models.gcn import GCN
from torch_geometric.nn.models import SchNet as SchNetPYG

# Config
TARGET       = 4
SEED         = 42
N_SAMPLES    = 2000
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GCN_CKPT     = 'logs/gcn/20260427_034205/best.pt'
SCHNET_CKPT  = 'logs/schnet/20260427_160158/best.pt'

torch.manual_seed(SEED)

# Load dataset
dataset = QM9(root='./data/QM9', transform=Distance())
mean = dataset._data.y[:, TARGET].mean().item()
std  = dataset._data.y[:, TARGET].std().item()

# Use val set subset
split    = np.load('data/split_42.npz')
val_idx  = split['val_idx'][:N_SAMPLES]
val_data = dataset[torch.tensor(val_idx)]
loader   = DataLoader(val_data, batch_size=64)

# Get gap values for coloring
gap_values = dataset._data.y[torch.tensor(val_idx), TARGET].numpy()

# ---- Extract GCN representations ----
class GCNWithHook(GCN):
    def forward_with_repr(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        return torch.cat([x_mean, x_max], dim=-1)

gcn = GCNWithHook().to(DEVICE)
gcn.load_state_dict(torch.load(GCN_CKPT, map_location=DEVICE))
gcn.eval()

gcn_reprs = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        gcn_reprs.append(gcn.forward_with_repr(batch).cpu())
gcn_reprs = torch.cat(gcn_reprs, dim=0).numpy()
print(f"GCN representations: {gcn_reprs.shape}")

# ---- Extract SchNet representations ----
class SchNetWithHook(SchNetPYG):
    def forward_with_repr(self, z, pos, batch):
        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        from torch_geometric.nn import global_add_pool
        return global_add_pool(h, batch)

schnet = SchNetWithHook(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=5.0,
    readout='add',
).to(DEVICE)
schnet.load_state_dict(torch.load(SCHNET_CKPT, map_location=DEVICE))
schnet.eval()

schnet_reprs = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        schnet_reprs.append(schnet.forward_with_repr(batch.z, batch.pos, batch.batch).cpu())
schnet_reprs = torch.cat(schnet_reprs, dim=0).numpy()
print(f"SchNet representations: {schnet_reprs.shape}")

# ---- t-SNE ----
print("Running t-SNE for GCN...")
gcn_tsne = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(gcn_reprs)

print("Running t-SNE for SchNet...")
schnet_tsne = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(schnet_reprs)

# ---- Plot GCN ----
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(gcn_tsne[:, 0], gcn_tsne[:, 1],
                c=gap_values, cmap='viridis', s=10, alpha=0.7)
ax.set_title('GCN Latent Space (t-SNE)')
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(sc, ax=ax, label='HOMO-LUMO Gap (eV)')
plt.tight_layout()
plt.savefig('logs/gcn_latent_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to logs/gcn_latent_tsne.png")

# ---- Plot SchNet ----
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(schnet_tsne[:, 0], schnet_tsne[:, 1],
                c=gap_values, cmap='viridis', s=10, alpha=0.7)
ax.set_title('SchNet Latent Space (t-SNE)')
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(sc, ax=ax, label='HOMO-LUMO Gap (eV)')
plt.tight_layout()
plt.savefig('logs/schnet_latent_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to logs/schnet_latent_tsne.png")