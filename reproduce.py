import torch
import numpy as np
import json
import csv
import os
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance
from torch_geometric.nn.models import SchNet as SchNetPYG
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import scatter
from sklearn.manifold import TSNE
from models.gcn import GCN

# ---- Config ----
TARGET = 4
SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Checkpoint paths
GCN_CKPT     = 'logs/gcn/20260429_135048/best.pt'
SCHNET_CKPT  = 'logs/schnet/20260429_192357/best.pt'
DIMENET_CKPT = '20260430_224522/20260430_224522/best.pt'

# Metrics CSV paths
GCN_CSV      = 'logs/gcn/20260429_135048/metrics.csv'
SCHNET_CSV   = 'logs/schnet/20260429_192357/metrics.csv'
DIMENET_CSV  = '20260430_224522/20260430_224522/metrics.csv'

# Learning curve paths
SCHNET_LC_JSON = 'logs/lc_schnet_results.json'
DIMENET_LC_CSV = 'Dimenet/dimenet_learning_curves/curve.csv'

os.makedirs('logs/reproduce', exist_ok=True)
torch.manual_seed(SEED)

# ---- Load dataset ----
print("\nLoading dataset...")
dataset = QM9(root='./data/QM9', transform=Distance())
mean = dataset._data.y[:, TARGET].mean().item()
std  = dataset._data.y[:, TARGET].std().item()

split        = np.load('data/split_42.npz')
test_dataset = dataset[torch.tensor(split['test_idx'])]
val_dataset  = dataset[torch.tensor(split['val_idx'])]
test_loader  = DataLoader(test_dataset, batch_size=32)
val_loader   = DataLoader(val_dataset,  batch_size=32)

# ---- Evaluation functions ----
@torch.no_grad()
def evaluate_gcn(model, loader):
    model.eval()
    mae = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch) * std + mean
        mae  += (pred - batch.y[:, TARGET]).abs().sum().item()
    return mae / len(loader.dataset)

@torch.no_grad()
def evaluate_geo(model, loader):
    model.eval()
    mae = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch.z, batch.pos, batch.batch).squeeze(-1) * std + mean
        mae  += (pred - batch.y[:, TARGET]).abs().sum().item()
    return mae / len(loader.dataset)

# ---- Load models ----
print("\nLoading models...")

gcn = GCN().to(DEVICE)
gcn.load_state_dict(torch.load(GCN_CKPT, map_location=DEVICE))

schnet = SchNetPYG(
    hidden_channels=128, num_filters=128, num_interactions=4,
    num_gaussians=50, cutoff=5.0, readout='add'
).to(DEVICE)
schnet.load_state_dict(torch.load(SCHNET_CKPT, map_location=DEVICE))

dimenet = DimeNetPlusPlus(
    hidden_channels=128, out_channels=1, num_blocks=4,
    int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
    num_spherical=7, num_radial=6, cutoff=5.0,
    envelope_exponent=5, num_before_skip=1,
    num_after_skip=2, num_output_layers=3,
).to(DEVICE)
dimenet.load_state_dict(torch.load(DIMENET_CKPT, map_location=DEVICE))

# ---- Table 1: Test MAE ----
print("\n=== Test MAE Results ===")
gcn_mae     = evaluate_gcn(gcn,     test_loader)
schnet_mae  = evaluate_geo(schnet,  test_loader)
dimenet_mae = evaluate_geo(dimenet, test_loader)
print(f"GCN:       {gcn_mae:.4f} eV")
print(f"SchNet:    {schnet_mae:.4f} eV")
print(f"DimeNet++: {dimenet_mae:.4f} eV")

results_summary = {
    'gcn_test_mae':     gcn_mae,
    'schnet_test_mae':  schnet_mae,
    'dimenet_test_mae': dimenet_mae,
}
with open('logs/reproduce/test_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# ---- Figure 1: Training Loss (MSE) ----
print("\nGenerating Figure 1: Training Loss...")

def load_metrics(path):
    train_mse, val_mae = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_mse.append(float(row['train_mse']))
            val_mae.append(float(row['val_mae']))
    return train_mse, val_mae

gcn_train,     gcn_val     = load_metrics(GCN_CSV)
schnet_train,  schnet_val  = load_metrics(SCHNET_CSV)
dimenet_train, dimenet_val = load_metrics(DIMENET_CSV)
epochs = list(range(1, 151))

plt.figure(figsize=(8, 5))
plt.plot(epochs, gcn_train,     label='GCN')
plt.plot(epochs, schnet_train,  label='SchNet')
plt.plot(epochs, dimenet_train, label='DimeNet++')
plt.xlabel('Epoch')
plt.ylabel('Train MSE')
plt.title('Training Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logs/reproduce/train_mse_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Figure 2: Validation MAE ----
print("Generating Figure 2: Validation MAE...")

plt.figure(figsize=(8, 5))
plt.plot(epochs, gcn_val,     label='GCN')
plt.plot(epochs, schnet_val,  label='SchNet')
plt.plot(epochs, dimenet_val, label='DimeNet++')
plt.xlabel('Epoch')
plt.ylabel('Val MAE (eV)')
plt.title('Validation MAE during Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logs/reproduce/val_mae_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Figure 3: Learning Curves ----
print("Generating Figure 3: Learning Curves...")

with open(SCHNET_LC_JSON) as f:
    schnet_lc = json.load(f)
schnet_sizes   = [int(k) for k in schnet_lc.keys()]
schnet_test_lc = [schnet_lc[k]['test_mae'] for k in schnet_lc.keys()]

dimenet_sizes, dimenet_test_lc = [], []
with open(DIMENET_LC_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        dimenet_sizes.append(int(row['n_train']))
        dimenet_test_lc.append(float(row['test_mae']))

plt.figure(figsize=(8, 5))
plt.plot(schnet_sizes,  schnet_test_lc,  'o-', label='SchNet')
plt.plot(dimenet_sizes, dimenet_test_lc, 's-', label='DimeNet++')
plt.xscale('log')
plt.xlabel('Training Set Size')
plt.ylabel('Test MAE (eV)')
plt.title('Learning Curves: SchNet vs DimeNet++')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logs/reproduce/learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Figures 4-6: t-SNE ----
print("Generating t-SNE visualizations...")

N_SAMPLES = 2000
val_idx   = split['val_idx'][:N_SAMPLES]
val_small = dataset[torch.tensor(val_idx)]
loader    = DataLoader(val_small, batch_size=64)
gap_vals  = dataset._data.y[torch.tensor(val_idx), TARGET].numpy()

# GCN representations
class GCNWithHook(GCN):
    def forward_with_repr(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
        return torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=-1)

# SchNet representations
class SchNetWithHook(SchNetPYG):
    def forward_with_repr(self, z, pos, batch):
        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        return global_add_pool(h, batch)

gcn_hook = GCNWithHook().to(DEVICE)
gcn_hook.load_state_dict(torch.load(GCN_CKPT, map_location=DEVICE))
gcn_hook.eval()

schnet_hook = SchNetWithHook(
    hidden_channels=128, num_filters=128, num_interactions=4,
    num_gaussians=50, cutoff=5.0, readout='add'
).to(DEVICE)
schnet_hook.load_state_dict(torch.load(SCHNET_CKPT, map_location=DEVICE))
schnet_hook.eval()

gcn_reprs, schnet_reprs = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        gcn_reprs.append(gcn_hook.forward_with_repr(batch).cpu())
        schnet_reprs.append(schnet_hook.forward_with_repr(
            batch.z, batch.pos, batch.batch).cpu())

gcn_reprs    = torch.cat(gcn_reprs,    dim=0).numpy()
schnet_reprs = torch.cat(schnet_reprs, dim=0).numpy()

# DimeNet++ representations via hook
dimenet_hook = DimeNetPlusPlus(
    hidden_channels=128, out_channels=1, num_blocks=4,
    int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
    num_spherical=7, num_radial=6, cutoff=5.0,
    envelope_exponent=5, num_before_skip=1,
    num_after_skip=2, num_output_layers=3,
).to(DEVICE)
dimenet_hook.load_state_dict(torch.load(DIMENET_CKPT, map_location=DEVICE))
dimenet_hook.eval()

captured = []
def hook_fn(module, inputs, output):
    captured.append(inputs[0])

hooks = [blk.lin.register_forward_hook(hook_fn) for blk in dimenet_hook.output_blocks]

dimenet_reprs = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        captured.clear()
        _ = dimenet_hook(batch.z, batch.pos, batch.batch)
        per_atom = torch.stack(captured, dim=0).sum(dim=0)
        per_mol  = scatter(per_atom, batch.batch, dim=0, reduce='sum')
        dimenet_reprs.append(per_mol.cpu())

for h in hooks:
    h.remove()

dimenet_reprs = torch.cat(dimenet_reprs, dim=0).numpy()

print("Running t-SNE for GCN...")
gcn_tsne     = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(gcn_reprs)
print("Running t-SNE for SchNet...")
schnet_tsne  = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(schnet_reprs)
print("Running t-SNE for DimeNet++...")
dimenet_tsne = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(dimenet_reprs)

for tsne, name, fname in [
    (gcn_tsne,     'GCN',       'gcn'),
    (schnet_tsne,  'SchNet',    'schnet'),
    (dimenet_tsne, 'DimeNet++', 'dimenet'),
]:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(tsne[:, 0], tsne[:, 1],
                    c=gap_vals, cmap='viridis', s=10, alpha=0.7)
    ax.set_title(f'{name} Latent Space (t-SNE)')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label='HOMO-LUMO Gap (eV)')
    plt.tight_layout()
    plt.savefig(f'logs/reproduce/{fname}_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\n=== All results reproduced ===")
print(f"GCN Test MAE:       {gcn_mae:.4f} eV")
print(f"SchNet Test MAE:    {schnet_mae:.4f} eV")
print(f"DimeNet++ Test MAE: {dimenet_mae:.4f} eV")
print("\nAll files saved to logs/reproduce/")
print("  - test_results.json")
print("  - train_mse_curve.png")
print("  - val_mae_curve.png")
print("  - learning_curve.png")
print("  - gcn_tsne.png")
print("  - schnet_tsne.png")
print("  - dimenet_tsne.png")