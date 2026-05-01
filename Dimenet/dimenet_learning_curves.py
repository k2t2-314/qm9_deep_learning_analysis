"""
Learning curve for DimeNet++ on the QM9 HOMO-LUMO gap task.

Trains DimeNet++ from scratch on increasing prefixes of the same
shuffled train_idx (so smaller-N runs are nested subsets of larger-N
runs). For each subset size we record best val MAE and final test MAE
into a CSV that you can later merge with GCN/SchNet curves to make the
figure for §4.4 of the report.

Usage:
    python dimenet_learning_curves.py
"""
import os
import csv
import time
import numpy as np
import torch
from datetime import datetime
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import DimeNetPlusPlus

TARGET       = 4
BATCH_SIZE   = 32
LR           = 1e-4
SEED         = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRAIN_LIST = [1000, 3000, 10000, 30000, 110831]
# Smaller subsets → more epochs (each epoch is fast, more headroom before
# overfitting); large subsets → fewer epochs to keep wall time reasonable.
EPOCHS_BY_N  = {1000: 200, 3000: 150, 10000: 120, 30000: 100, 110831: 80}

torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

dataset = QM9(root='./data/QM9')
mean = dataset._data.y[:, TARGET].mean().item()
std  = dataset._data.y[:, TARGET].std().item()
print(f"Target mean: {mean:.4f}, std: {std:.4f}")

split         = np.load('data/split_42.npz')
all_train_idx = split['train_idx']
val_dataset   = dataset[torch.tensor(split['val_idx'])]
test_dataset  = dataset[torch.tensor(split['test_idx'])]

val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir   = os.path.join('logs', 'dimenet_pp_curves', timestamp)
os.makedirs(out_dir, exist_ok=True)
csv_path  = os.path.join(out_dir, 'curve.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['model', 'n_train', 'epochs', 'best_val_mae',
                            'test_mae', 'test_mse', 'wall_seconds'])
print(f"Logging to {csv_path}")


def build_model():
    return DimeNetPlusPlus(
        hidden_channels=128, out_channels=1, num_blocks=4,
        int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
        num_spherical=7, num_radial=6, cutoff=5.0,
        envelope_exponent=5, num_before_skip=1,
        num_after_skip=2, num_output_layers=3,
    ).to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mae = mse = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch.z, batch.pos, batch.batch).squeeze(-1) * std + mean
        mae  += (pred - batch.y[:, TARGET]).abs().sum().item()
        mse  += ((pred - batch.y[:, TARGET]) ** 2).sum().item()
    n = len(loader.dataset)
    return mae / n, mse / n


for n_train in N_TRAIN_LIST:
    print(f"\n=== n_train = {n_train} ===")
    sub_idx       = all_train_idx[:n_train]   # nested prefix, fair across sizes
    train_dataset = dataset[torch.tensor(sub_idx)]
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    loss_fn = torch.nn.MSELoss()

    epochs       = EPOCHS_BY_N[n_train]
    best_val_mae = float('inf')
    best_state   = None
    t0           = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch  = batch.to(DEVICE)
            target = (batch.y[:, TARGET] - mean) / std
            optimizer.zero_grad()
            pred = model(batch.z, batch.pos, batch.batch).squeeze(-1)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

        val_mae, _ = evaluate(model, val_loader)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:03d}  val MAE {val_mae:.4f}  best {best_val_mae:.4f}")

    model.load_state_dict(best_state)
    test_mae, test_mse = evaluate(model, test_loader)
    wall = time.time() - t0
    print(f"  n_train={n_train}  test MAE {test_mae:.4f}  ({wall/60:.1f} min)")

    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(['dimenet_pp', n_train, epochs, best_val_mae,
                                test_mae, test_mse, wall])

print(f"\nLearning curve CSV: {csv_path}")
