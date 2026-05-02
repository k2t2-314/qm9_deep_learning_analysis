"""Contribution 1 + 3: latent-space visualization and error analysis.

This script loads a trained checkpoint, extracts graph-level latent embeddings,
and analyzes which molecules have the largest prediction errors.

Outputs:
- analysis_outputs/latent_space_error_pca.png
- analysis_outputs/latent_space_num_atoms_pca.png
- analysis_outputs/error_vs_num_atoms.png
- analysis_outputs/mean_error_by_atom_count.png
- analysis_outputs/error_by_has_N.png
- analysis_outputs/error_by_has_O.png
- analysis_outputs/error_by_has_F.png
- analysis_outputs/test_error_analysis.csv
- analysis_outputs/worst_20_predictions.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from qm9_models import GCNRegressor
from qm9_utils import (
    set_seed, get_device, load_qm9_dataset, make_random_split, make_loaders,
    get_target, denormalize_y, ensure_dir
)


ATOM_NAMES = ["H", "C", "N", "O", "F"]


def load_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = GCNRegressor(
        in_dim=ckpt.get("in_dim", 11),
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        out_dim=1,
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_predictions_embeddings(model, loader, device, target_idx, y_mean, y_std, test_indices=None):
    embeddings = []
    errors = []
    records = []
    running_graph_id = 0

    for data in loader:
        data = data.to(device)
        pred_norm, emb = model(data, return_embedding=True)
        y = get_target(data, target_idx).to(device)
        pred_real = denormalize_y(pred_norm, y_mean, y_std)
        abs_error = torch.abs(pred_real - y).view(-1)

        batch = data.batch
        atom_counts = torch.bincount(batch, minlength=data.num_graphs)

        embeddings.append(emb.cpu())
        errors.append(abs_error.cpu())

        for i in range(data.num_graphs):
            molecule_mask = batch == i
            atom_features = data.x[molecule_mask].detach().cpu()

            # PyG QM9 uses the first 5 dimensions as one-hot atom type features for H, C, N, O, F.
            # If your preprocessing is different, adjust this block.
            atom_type_counts = atom_features[:, :5].sum(dim=0).numpy()
            atom_count_dict = {f"num_{name}": int(atom_type_counts[j]) for j, name in enumerate(ATOM_NAMES)}
            atom_presence_dict = {f"has_{name}": bool(atom_type_counts[j] > 0) for j, name in enumerate(ATOM_NAMES)}

            dataset_index = None
            if test_indices is not None:
                dataset_index = int(test_indices[running_graph_id].item())

            record = {
                "local_test_id": running_graph_id,
                "dataset_index": dataset_index,
                "true": float(y[i].item()),
                "pred": float(pred_real[i].item()),
                "abs_error": float(abs_error[i].item()),
                "num_atoms": int(atom_counts[i].item()),
                "num_bonds_directed_edges": int((batch[data.edge_index[0]] == i).sum().item()),
            }
            record.update(atom_count_dict)
            record.update(atom_presence_dict)
            records.append(record)
            running_graph_id += 1

    embeddings = torch.cat(embeddings, dim=0).numpy()
    errors = torch.cat(errors, dim=0).numpy()
    df = pd.DataFrame(records)
    return embeddings, errors, df


def plot_latent_space(embeddings, df, output_dir):
    pca = PCA(n_components=2)
    z = pca.fit_transform(embeddings)
    df["pca1"] = z[:, 0]
    df["pca2"] = z[:, 1]

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df["pca1"], df["pca2"], c=df["abs_error"], s=8, alpha=0.7)
    plt.colorbar(sc, label="Absolute Error")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Space Colored by Prediction Error")
    plt.tight_layout()
    path1 = os.path.join(output_dir, "latent_space_error_pca.png")
    plt.savefig(path1, dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df["pca1"], df["pca2"], c=df["num_atoms"], s=8, alpha=0.7)
    plt.colorbar(sc, label="Number of Atoms")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Space Colored by Molecular Size")
    plt.tight_layout()
    path2 = os.path.join(output_dir, "latent_space_num_atoms_pca.png")
    plt.savefig(path2, dpi=300)
    plt.close()

    return path1, path2, df


def plot_error_analysis(df, output_dir):
    paths = []

    plt.figure(figsize=(7, 5))
    plt.scatter(df["num_atoms"], df["abs_error"], s=10, alpha=0.5)
    plt.xlabel("Number of Atoms")
    plt.ylabel("Absolute Error")
    plt.title("Prediction Error vs Molecular Size")
    plt.tight_layout()
    p = os.path.join(output_dir, "error_vs_num_atoms.png")
    plt.savefig(p, dpi=300)
    plt.close()
    paths.append(p)

    grouped = df.groupby("num_atoms", as_index=False)["abs_error"].mean()
    plt.figure(figsize=(7, 5))
    plt.plot(grouped["num_atoms"], grouped["abs_error"], marker="o")
    plt.xlabel("Number of Atoms")
    plt.ylabel("Mean Absolute Error")
    plt.title("Mean Error by Molecular Size")
    plt.tight_layout()
    p = os.path.join(output_dir, "mean_error_by_atom_count.png")
    plt.savefig(p, dpi=300)
    plt.close()
    paths.append(p)

    for atom in ["N", "O", "F"]:
        col = f"has_{atom}"
        plt.figure(figsize=(6, 5))
        df.boxplot(column="abs_error", by=col)
        plt.title(f"Error Distribution: Molecules With vs Without {atom}")
        plt.suptitle("")
        plt.xlabel(f"Contains {atom}")
        plt.ylabel("Absolute Error")
        plt.tight_layout()
        p = os.path.join(output_dir, f"error_by_has_{atom}.png")
        plt.savefig(p, dpi=300)
        plt.close()
        paths.append(p)

    return paths


def summarize_error_groups(df, output_dir):
    rows = []
    for col in ["has_N", "has_O", "has_F"]:
        summary = df.groupby(col)["abs_error"].agg(["count", "mean", "median", "std"]).reset_index()
        summary.insert(0, "group", col)
        summary = summary.rename(columns={col: "value"})
        rows.append(summary)

    group_summary = pd.concat(rows, ignore_index=True)
    path = os.path.join(output_dir, "error_group_summary.csv")
    group_summary.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/QM9")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_baseline.pt")
    parser.add_argument("--output_dir", type=str, default="./analysis_outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = get_device()
    print("Device:", device)

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    target_idx = int(ckpt.get("target_idx", 4))
    y_mean = ckpt["y_mean"].to(device).float()
    y_std = ckpt["y_std"].to(device).float()

    dataset = load_qm9_dataset(args.root)
    train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx = make_random_split(
        dataset, 110831, 10000, 10000, seed=args.seed
    )
    _, _, test_loader = make_loaders(train_dataset, val_dataset, test_dataset, args.batch_size, args.num_workers)

    embeddings, errors, df = collect_predictions_embeddings(
        model, test_loader, device, target_idx, y_mean, y_std, test_indices=test_idx
    )

    latent1, latent2, df = plot_latent_space(embeddings, df, args.output_dir)
    error_plot_paths = plot_error_analysis(df, args.output_dir)
    group_summary_path = summarize_error_groups(df, args.output_dir)

    full_csv = os.path.join(args.output_dir, "test_error_analysis.csv")
    df.to_csv(full_csv, index=False)

    worst_csv = os.path.join(args.output_dir, "worst_20_predictions.csv")
    worst20 = df.sort_values("abs_error", ascending=False).head(20)
    worst20.to_csv(worst_csv, index=False)

    print("Saved latent plots:")
    print(" ", latent1)
    print(" ", latent2)
    print("Saved error plots:")
    for p in error_plot_paths:
        print(" ", p)
    print("Saved CSV:", full_csv)
    print("Saved CSV:", worst_csv)
    print("Saved CSV:", group_summary_path)
    print("\nWorst 20 predictions:")
    print(worst20[["local_test_id", "dataset_index", "true", "pred", "abs_error", "num_atoms", "num_C", "num_N", "num_O", "num_F"]])


if __name__ == "__main__":
    main()
