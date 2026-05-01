import torch
import torch.nn as nn
import json
import os
import csv
import numpy as np
from datetime import datetime
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance
from torch_geometric.nn.models import SchNet as SchNetPYG

TARGET     = 4
BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 150
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)

dataset = QM9(root='./data/QM9', transform=Distance())

mean = dataset._data.y[:, TARGET].mean().item()
std  = dataset._data.y[:, TARGET].std().item()
print(f"Target mean: {mean:.4f}, std: {std:.4f}")

split         = np.load('data/split_42.npz')
train_dataset = dataset[torch.tensor(split['train_idx'])]
val_dataset   = dataset[torch.tensor(split['val_idx'])]
test_dataset  = dataset[torch.tensor(split['test_idx'])]
print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir   = os.path.join('logs', 'schnet', timestamp)
os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)

# Model
model = SchNetPYG(
    hidden_channels=128,
    num_filters=128,
    num_interactions=4,
    num_gaussians=50,
    cutoff=5.0,
    readout='add',
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15
)
loss_fn = torch.nn.MSELoss()

def train_epoch():
    model.train()
    total_mse = 0
    for batch in train_loader:
        batch  = batch.to(DEVICE)
        target = (batch.y[:, TARGET] - mean) / std
        optimizer.zero_grad()
        pred = model(batch.z, batch.pos, batch.batch).squeeze(-1)
        loss   = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_mse += loss.item() * batch.num_graphs
    return total_mse / len(train_loader.dataset)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    mae = 0
    mse = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch.z, batch.pos, batch.batch).squeeze(-1) * std + mean
        mae  += (pred - batch.y[:, TARGET]).abs().sum().item()
        mse  += ((pred - batch.y[:, TARGET]) ** 2).sum().item()
    n = len(loader.dataset)
    return mae / n, mse / n

# CSV logger
csv_path = os.path.join(run_dir, 'metrics.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_mse', 'val_mse', 'val_mae'])

best_val_mae = float('inf')

print(f"\nTraining SchNet (PyG)...")
for epoch in range(1, EPOCHS + 1):
    train_mse        = train_epoch()
    val_mae, val_mse = evaluate(val_loader)
    scheduler.step(val_mae)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_mse, val_mse, val_mae])

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))

    if epoch % 10 == 0:
        ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch:03d}.pt')
        torch.save(model.state_dict(), ckpt_path)
    print(f"Epoch {epoch:03d} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} eV")

# Test
model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt')))
test_mae, test_mse = evaluate(test_loader)
print(f"\nSchNet Test MAE: {test_mae:.4f} eV | Test MSE: {test_mse:.4f}")

results = {
    'model':        'schnet_pyg',
    'epochs':       EPOCHS,
    'lr':           LR,
    'seed':         SEED,
    'best_val_mae': best_val_mae,
    'test_mae':     test_mae,
    'test_mse':     test_mse,
}
with open(os.path.join(run_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {run_dir}")