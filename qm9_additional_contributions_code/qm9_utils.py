import os
import random
import numpy as np
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_qm9_dataset(root="./data/QM9"):
    return QM9(root=root)


def make_random_split(dataset, train_size=110831, val_size=10000, test_size=10000, seed=42):
    total = train_size + val_size + test_size
    if total > len(dataset):
        raise ValueError(f"Requested split size {total}, but dataset only has {len(dataset)} samples.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g)[:total]

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:train_size + val_size + test_size]

    return dataset[train_idx], dataset[val_idx], dataset[test_idx], train_idx, val_idx, test_idx


def compute_target_stats(train_dataset, target_idx=4):
    ys = []
    for data in train_dataset:
        ys.append(data.y[:, target_idx].view(-1))
    y = torch.cat(ys).float()
    mean = y.mean()
    std = y.std()
    if std.item() == 0:
        raise ValueError("Target std is zero. Cannot normalize target.")
    return mean, std


def make_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_target(data, target_idx=4):
    return data.y[:, target_idx].view(-1, 1).float()


def normalize_y(y, y_mean, y_std):
    return (y - y_mean.to(y.device)) / y_std.to(y.device)


def denormalize_y(y_norm, y_mean, y_std):
    return y_norm * y_std.to(y_norm.device) + y_mean.to(y_norm.device)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
