# QM9 HOMO-LUMO Gap Prediction

This repository contains the code for our deep learning mini-project (24-788, Spring 2026).
We compare graph neural network models of increasing geometric awareness for predicting
the HOMO-LUMO energy gap on the QM9 molecular dataset.

## Models

- **GCN**: Graph Convolutional Network (baseline, topology only)
- **GCN with Distance**: GCN with scalar distance edge weighting (ablation)
- **SchNet**: Continuous-filter convolutional network using interatomic distances
- **DimeNet++**: Directional message passing network using distances and bond angles

## Results

| Model | Val MAE (eV) | Test MAE (eV) |
|-------|-------------|---------------|
| GCN | 0.1440 | 0.1449 |
| GCN + Distance | 0.1449 | 0.1444 |
| SchNet | 0.0682 | 0.0681 |
| DimeNet++ | 0.0484 | 0.0487 |
| SOTA | < 0.050 | < 0.050 |

## Requirements

```bash
conda create -n qm9 python=3.10
conda activate qm9
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install scikit-learn matplotlib numpy
```

## Dataset

QM9 is downloaded automatically via PyTorch Geometric on first run.
We predict target index 4 (HOMO-LUMO gap, in eV).

## Data Split

Generate the fixed train/val/test split before training:

```bash
python split_dataset.py
```

This creates `data/split_42.npz` with:
- Train: 110,831 molecules
- Val: 11,000 molecules
- Test: 11,000 molecules
- Random seed: 42

## Training

```bash
# Train GCN and GCN with Distance
python train_gcn.py

# Train SchNet
python train_schnet.py

# Train DimeNet++
python train_dimenet.py
```

## Training Configuration

All models share the same hyperparameters for fair comparison:

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 150 |
| Optimizer | Adam (weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=15) |
| Loss | MSELoss |
| Random seed | 42 |

## Logging

Each training run saves results to `logs/<model>/<timestamp>/`:
- `metrics.csv`: train_mse, val_mse, val_mae per epoch
- `best.pt`: model checkpoint with best validation MAE
- `results.json`: final test MAE and training configuration
- `checkpoints/`: model checkpoint every 10 epochs

## Reproducing Results

All key results and figures from the report can be reproduced without retraining:

```bash
conda activate qm9
python reproduce.py
```

This script will:
1. Load trained model checkpoints
2. Evaluate all models on the test set and print Test MAE
3. Save all figures to `logs/reproduce/`:
   - `test_results.json`: Test MAE for all models
   - `train_mse_curve.png`: Training loss curves
   - `val_mae_curve.png`: Validation MAE curves
   - `learning_curve.png`: Learning curves (SchNet vs DimeNet++)
   - `gcn_tsne.png`: GCN latent space visualization
   - `schnet_tsne.png`: SchNet latent space visualization
   - `dimenet_tsne.png`: DimeNet++ latent space visualization

## Analysis Scripts

```bash
# Latent space visualization (GCN and SchNet)
python analyze_latent.py

# DimeNet++ latent space visualization
python extract_dimenet_latents.py --ckpt 20260430_224522/20260430_224522/best.pt

# SchNet learning curves
python schnet_learning_curves.py

# Plot training curves
python plot_loss.py
```

## Repository Structure

```
qm9-project/
├── models/
│   ├── gcn.py              # GCN model
│   └── gcn_dist.py         # GCN with distance weighting
├── logs/
│   ├── gcn/                # GCN training logs and checkpoints
│   ├── schnet/             # SchNet training logs and checkpoints
│   └── reproduce/          # Reproduced figures and results
├── 20260430_224522/        # DimeNet++ training logs and checkpoints
├── Dimenet/                # DimeNet++ learning curve results
├── split_dataset.py        # Generate data split
├── train_gcn.py            # Train GCN and GCN+Distance
├── train_schnet.py         # Train SchNet
├── train_dimenet.py        # Train DimeNet++
├── reproduce.py            # Reproduce all results and figures
├── analyze_latent.py       # t-SNE for GCN and SchNet
├── extract_dimenet_latents.py  # t-SNE for DimeNet++
├── schnet_learning_curves.py   # SchNet learning curves
├── plot_loss.py            # Plot training curves
├── explore_data.py         # Dataset exploration
└── split_dataset.py        # Generate train/val/test split
```
