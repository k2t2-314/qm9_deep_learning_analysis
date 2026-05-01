import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance
from torch_geometric.nn.models import SchNet as SchNetPYG

# Config
TARGET      = 4
BATCH_SIZE  = 32
LR          = 1e-4
EPOCHS      = 50
SEED        = 42
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_SIZES = [1000, 3000, 10000, 30000, 110831]
EPOCHS_BY_N = {1000: 200, 3000: 150, 10000: 120, 30000: 100, 110831: 80}

torch.manual_seed(SEED)

dataset = QM9(root='./data/QM9', transform=Distance())
mean = dataset._data.y[:, TARGET].mean().item()
std  = dataset._data.y[:, TARGET].std().item()

split        = np.load('data/split_42.npz')
val_dataset  = dataset[torch.tensor(split['val_idx'])]
test_dataset = dataset[torch.tensor(split['test_idx'])]
val_loader   = DataLoader(val_dataset,  batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def build_model():
    return SchNetPYG(
        hidden_channels=128,
        num_filters=128,
        num_interactions=4,
        num_gaussians=50,
        cutoff=5.0,
        readout='add',
    ).to(DEVICE)

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_mse = 0
    for batch in loader:
        batch  = batch.to(DEVICE)
        target = (batch.y[:, TARGET] - mean) / std
        optimizer.zero_grad()
        pred   = model(batch.z, batch.pos, batch.batch).squeeze(-1)
        loss   = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_mse += loss.item() * batch.num_graphs
    return total_mse / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mae = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch.z, batch.pos, batch.batch).squeeze(-1) * std + mean
        mae  += (pred - batch.y[:, TARGET]).abs().sum().item()
    return mae / len(loader.dataset)

results = {}

for n_train in TRAIN_SIZES:
    print(f"\nTraining with {n_train} samples...")
    torch.manual_seed(SEED)

    train_subset = dataset[torch.tensor(split['train_idx'][:n_train])]
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    model     = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    loss_fn = torch.nn.MSELoss()

    best_val_mae = float('inf')
    for epoch in range(1, EPOCHS_BY_N[n_train] + 1):
        train_epoch(model, train_loader, optimizer, loss_fn)
        val_mae = evaluate(model, val_loader)
        scheduler.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f'logs/lc_schnet_{n_train}.pt')
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Val MAE: {val_mae:.4f} eV")

    model.load_state_dict(torch.load(f'logs/lc_schnet_{n_train}.pt'))
    test_mae = evaluate(model, test_loader)
    results[n_train] = {'val_mae': best_val_mae, 'test_mae': test_mae}
    print(f"  Best Val MAE: {best_val_mae:.4f} | Test MAE: {test_mae:.4f}")

# Plot
train_sizes   = list(results.keys())
schnet_test   = [results[n]['test_mae'] for n in train_sizes]
dimenet_test  = [0.2999, 0.1925, 0.1248, 0.0851, 0.0505]

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, schnet_test,  'o-', label='SchNet')
plt.plot(train_sizes, dimenet_test, 's-', label='DimeNet++')
plt.xscale('log')
plt.xlabel('Training Set Size')
plt.ylabel('Test MAE (eV)')
plt.title('Learning Curves: SchNet vs DimeNet++')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logs/learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to logs/learning_curve.png")

with open('logs/lc_schnet_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved.")