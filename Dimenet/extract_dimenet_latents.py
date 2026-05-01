"""
Extract per-molecule latent embeddings from a trained DimeNet++ model
and save a t-SNE visualization to image/DimeNet_latent.png.

Latent definition: in each OutputPPBlock the per-atom features just
before the final scalar projection (`blk.lin`) are captured via a
forward hook, summed across all output blocks, then summed (scatter)
over atoms to the molecule level. The latent dim is `out_emb_channels`
(=256 in our config).

Usage:
    python extract_dimenet_latents.py --ckpt logs/dimenet_pp/<run>/best.pt
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.utils import scatter
from sklearn.manifold import TSNE

TARGET     = 4
BATCH_SIZE = 32
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, help='Path to best.pt')
parser.add_argument('--n_samples', type=int, default=5000,
                    help='Number of test molecules to embed (TSNE on full test is slow)')
parser.add_argument('--out', default='image/DimeNet_latent.png')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

# Data: same split as training, sample a subset of the test set for t-SNE
dataset  = QM9(root='./data/QM9')
split    = np.load('data/split_42.npz')
test_idx = split['test_idx']
rng      = np.random.default_rng(0)
n_pick   = min(args.n_samples, len(test_idx))
sub_idx  = rng.choice(test_idx, size=n_pick, replace=False)
sub_dataset = dataset[torch.tensor(sub_idx)]
loader      = DataLoader(sub_dataset, batch_size=BATCH_SIZE)
print(f"Embedding {n_pick} test molecules")

# Model: must match train_dimenet.py exactly so state_dict loads
model = DimeNetPlusPlus(
    hidden_channels=128, out_channels=1, num_blocks=4,
    int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
    num_spherical=7, num_radial=6, cutoff=5.0,
    envelope_exponent=5, num_before_skip=1,
    num_after_skip=2, num_output_layers=3,
).to(DEVICE)
model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
model.eval()

# Hook the input of every OutputPPBlock.lin (the final scalar projection).
# Each captured tensor is [num_atoms_in_batch, out_emb_channels=256].
captured = []
def hook(module, inputs, output):
    captured.append(inputs[0])

hooks = [blk.lin.register_forward_hook(hook) for blk in model.output_blocks]

embeds, labels = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        captured.clear()
        _ = model(batch.z, batch.pos, batch.batch)
        # sum across output blocks -> per-atom 256-d, then scatter sum -> per-molecule
        per_atom = torch.stack(captured, dim=0).sum(dim=0)
        per_mol  = scatter(per_atom, batch.batch, dim=0, reduce='sum')
        embeds.append(per_mol.cpu().numpy())
        labels.append(batch.y[:, TARGET].cpu().numpy())

for h in hooks:
    h.remove()

embeds = np.vstack(embeds)
labels = np.concatenate(labels)
print(f"Embeddings: {embeds.shape}")

print("Running t-SNE...")
emb_2d = TSNE(n_components=2, perplexity=30, init='pca',
              random_state=0).fit_transform(embeds)

plt.figure(figsize=(7, 6))
sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, s=3, cmap='viridis')
plt.colorbar(sc, label='HOMO-LUMO gap (eV)')
plt.xticks([]); plt.yticks([])
plt.title('DimeNet latent space (t-SNE)')
plt.tight_layout()
plt.savefig(args.out, dpi=200, bbox_inches='tight')
print(f"Saved {args.out}")
