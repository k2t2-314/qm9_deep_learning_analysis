"""Contribution 2: QM9 model-capacity ablation.

This script trains three GCN variants while keeping the training setup fixed:
Small:    2 layers, hidden dim 64
Baseline: 3 layers, hidden dim 128
Large:    5 layers, hidden dim 256

Outputs:
- checkpoints/best_small.pt
- checkpoints/best_baseline.pt
- checkpoints/best_large.pt
- results/capacity_ablation_results.csv
- results/training_curves.png
"""

import os
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from qm9_models import GCNRegressor, count_parameters
from qm9_utils import (
    set_seed, get_device, load_qm9_dataset, make_random_split, compute_target_stats,
    make_loaders, get_target, normalize_y, denormalize_y, ensure_dir
)


@torch.no_grad()
def evaluate(model, loader, device, target_idx, y_mean, y_std):
    model.eval()
    total_abs_error = 0.0
    total_sq_error_norm = 0.0
    n = 0

    for data in loader:
        data = data.to(device)
        pred_norm = model(data)
        y = get_target(data, target_idx).to(device)
        y_norm = normalize_y(y, y_mean, y_std)

        pred_real = denormalize_y(pred_norm, y_mean, y_std)
        abs_error = torch.abs(pred_real - y)
        total_abs_error += abs_error.sum().item()

        mse_norm = ((pred_norm - y_norm) ** 2).sum().item()
        total_sq_error_norm += mse_norm
        n += y.shape[0]

    mae = total_abs_error / n
    mse_norm = total_sq_error_norm / n
    return mae, mse_norm


def train_one_model(cfg, train_loader, val_loader, test_loader, device, args, y_mean, y_std):
    model = GCNRegressor(
        in_dim=args.in_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        out_dim=1,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.factor, patience=args.patience
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_epoch = -1
    history = []
    ckpt_path = os.path.join(args.checkpoint_dir, f"best_{cfg['name']}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            pred_norm = model(data)
            y = get_target(data, args.target_idx).to(device)
            y_norm = normalize_y(y, y_mean, y_std)

            loss = criterion(pred_norm, y_norm)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.shape[0]
            n += y.shape[0]

        train_mse = total_loss / n
        val_mae, val_mse_norm = evaluate(model, val_loader, device, args.target_idx, y_mean, y_std)
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_mse_norm": train_mse,
            "val_mse_norm": val_mse_norm,
            "val_mae_original_units": val_mae,
            "lr": current_lr,
        }
        history.append(row)

        print(
            f"[{cfg['name']}] Epoch {epoch:03d} | "
            f"train MSE(norm)={train_mse:.6f} | val MAE={val_mae:.6f} | lr={current_lr:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "target_idx": args.target_idx,
                    "y_mean": y_mean.cpu(),
                    "y_std": y_std.cpu(),
                    "best_val_mae": best_val_mae,
                    "best_epoch": best_epoch,
                    "in_dim": args.in_dim,
                    "dropout": args.dropout,
                    "seed": args.seed,
                },
                ckpt_path,
            )

    # Load best checkpoint and evaluate test set.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_mae, test_mse_norm = evaluate(model, test_loader, device, args.target_idx, y_mean, y_std)

    hist_path = os.path.join(args.result_dir, f"history_{cfg['name']}.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)

    return {
        "model": cfg["name"],
        "num_layers": cfg["num_layers"],
        "hidden_dim": cfg["hidden_dim"],
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_mae_original_units": best_val_mae,
        "test_mae_original_units": test_mae,
        "test_mse_norm": test_mse_norm,
        "checkpoint": ckpt_path,
        "history_csv": hist_path,
    }


def plot_training_curves(result_dir):
    plt.figure(figsize=(7, 5))
    for filename in sorted(os.listdir(result_dir)):
        if filename.startswith("history_") and filename.endswith(".csv"):
            name = filename.replace("history_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(result_dir, filename))
            plt.plot(df["epoch"], df["val_mae_original_units"], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE, original units")
    plt.title("Capacity Ablation: Validation MAE")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(result_dir, "training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/QM9")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--target_idx", type=int, default=4, help="QM9 target index. 4 = HOMO-LUMO gap.")
    parser.add_argument("--in_dim", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.result_dir)
    ensure_dir(args.checkpoint_dir)
    device = get_device()
    print("Device:", device)

    dataset = load_qm9_dataset(args.root)
    train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx = make_random_split(
        dataset, 110831, 10000, 10000, seed=args.seed
    )
    y_mean, y_std = compute_target_stats(train_dataset, args.target_idx)
    print(f"Target idx={args.target_idx}, train mean={y_mean.item():.6f}, train std={y_std.item():.6f}")

    split_info = {
        "seed": args.seed,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "target_idx": args.target_idx,
        "y_mean": float(y_mean.item()),
        "y_std": float(y_std.item()),
    }
    with open(os.path.join(args.result_dir, "split_and_target_stats.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    train_loader, val_loader, test_loader = make_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    configs = [
        {"name": "small", "hidden_dim": 64, "num_layers": 2},
        {"name": "baseline", "hidden_dim": 128, "num_layers": 3},
        {"name": "large", "hidden_dim": 256, "num_layers": 5},
    ]

    results = []
    for cfg in configs:
        results.append(train_one_model(cfg, train_loader, val_loader, test_loader, device, args, y_mean, y_std))

    results_df = pd.DataFrame(results)
    results_csv = os.path.join(args.result_dir, "capacity_ablation_results.csv")
    results_df.to_csv(results_csv, index=False)
    curve_path = plot_training_curves(args.result_dir)

    print("\nFinal ablation results:")
    print(results_df)
    print("Saved:", results_csv)
    print("Saved:", curve_path)


if __name__ == "__main__":
    main()
