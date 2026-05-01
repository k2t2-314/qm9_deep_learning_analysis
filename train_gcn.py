import torch
import json
import os
import csv
import numpy as np
from datetime import datetime
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance
from models.gcn import GCN
from models.gcn_dist import GCNDist

# Config
TARGET     = 4
BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 150
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)

# Load dataset
dataset = QM9(root='./data/QM9', transform=Distance())

# Normalize target
mean = dataset._data.y[:, TARGET].mean()
std  = dataset._data.y[:, TARGET].std()
print(f"Target mean: {mean:.4f}, std: {std:.4f}")

# Load split
split     = np.load('data/split_42.npz')
train_idx = split['train_idx']
val_idx   = split['val_idx']
test_idx  = split['test_idx']

train_dataset = dataset[torch.tensor(train_idx)]
val_dataset   = dataset[torch.tensor(val_idx)]
test_dataset  = dataset[torch.tensor(test_idx)]
print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

def make_run_dir(model_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = os.path.join('logs', model_name, timestamp)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    return run_dir

def train_epoch(model, optimizer, loss_fn):
    model.train()
    total_mse = 0
    for batch in train_loader:
        batch  = batch.to(DEVICE)
        target = (batch.y[:, TARGET] - mean) / std
        optimizer.zero_grad()
        pred   = model(batch)
        loss   = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_mse += loss.item() * batch.num_graphs
    return total_mse / len(train_loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mae = 0
    mse = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch) * std + mean
        mae  += (pred - batch.y[:, TARGET]).abs().sum().item()
        mse  += ((pred - batch.y[:, TARGET]) ** 2).sum().item()
    n = len(loader.dataset)
    return mae / n, mse / n

def train_model(model, model_name):
    run_dir   = make_run_dir(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    loss_fn = torch.nn.MSELoss()

    best_val_mae = float('inf')
    records      = []

    # CSV logger
    csv_path = os.path.join(run_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_mse', 'val_mse', 'val_mae'])

    print(f"\nTraining {model_name}...")
    for epoch in range(1, EPOCHS + 1):
        train_mse        = train_epoch(model, optimizer, loss_fn)
        val_mae, val_mse = evaluate(model, val_loader)
        scheduler.step(val_mae)

        records.append({
            'epoch':     epoch,
            'train_mse': train_mse,
            'val_mse':   val_mse,
            'val_mae':   val_mae
        })

        # CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_mse, val_mse, val_mae])

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))

        # Checkpoint every epoch
        if epoch % 10 == 0:
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch:03d}.pt')
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch:03d} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f} eV")

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pt')))
    test_mae, test_mse = evaluate(model, test_loader)
    print(f"\n{model_name} Test MAE: {test_mae:.4f} eV | Test MSE: {test_mse:.4f}")

    # Save results
    results = {
        'model':        model_name,
        'epochs':       EPOCHS,
        'lr':           LR,
        'seed':         SEED,
        'best_val_mae': best_val_mae,
        'test_mae':     test_mae,
        'test_mse':     test_mse,
        'records':      records
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {run_dir}")
    return run_dir

# Train GCN
model_gcn = GCN().to(DEVICE)
train_model(model_gcn, 'gcn')

# Train GCN+dist
model_gcn_dist = GCNDist().to(DEVICE)
train_model(model_gcn_dist, 'gcn_dist')