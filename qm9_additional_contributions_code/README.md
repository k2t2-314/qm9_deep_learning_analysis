# QM9 Molecular Property Prediction Contributions

This code supports three project contributions for the QM9 molecular property prediction task.

## Contributions

1. **Latent-space reconstruction**  
   Load a trained GCN checkpoint, extract the graph-level molecular embedding before the final regression layer, and visualize the learned latent space with PCA.

2. **Model-capacity ablation**  
   Train three GCN variants while keeping the optimization setup fixed:
   - Small: 2 layers, hidden dim 64
   - Baseline: 3 layers, hidden dim 128
   - Large: 5 layers, hidden dim 256

3. **Failure-case / error analysis**  
   Load the best checkpoint and analyze which molecules have the largest prediction errors by molecule size and atom composition.

## Environment

Install PyTorch and PyTorch Geometric according to your CUDA version. On Colab, a typical setup is:

```bash
pip install torch-geometric
pip install pandas scikit-learn matplotlib
```

If PyTorch Geometric installation fails, use the official installation selector:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Dataset

The code uses PyTorch Geometric's built-in QM9 loader:

```python
from torch_geometric.datasets import QM9
dataset = QM9(root='./data/QM9')
```

The split is:

- Train: 110831
- Validation: 10000
- Test: 10000
- Random seed: 42

The default target is index 4, the HOMO-LUMO gap.

## Training the capacity ablation

```bash
python train_capacity_ablation.py \
  --root ./data/QM9 \
  --target_idx 4 \
  --batch_size 32 \
  --epochs 150 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --factor 0.5 \
  --patience 15 \
  --seed 42
```

Outputs:

```text
checkpoints/best_small.pt
checkpoints/best_baseline.pt
checkpoints/best_large.pt
results/capacity_ablation_results.csv
results/training_curves.png
```

## Running latent-space and error analysis

Use the baseline checkpoint by default:

```bash
python analyze_best_checkpoint.py \
  --root ./data/QM9 \
  --checkpoint ./checkpoints/best_baseline.pt \
  --output_dir ./analysis_outputs \
  --batch_size 32 \
  --seed 42
```

Outputs:

```text
analysis_outputs/latent_space_error_pca.png
analysis_outputs/latent_space_num_atoms_pca.png
analysis_outputs/error_vs_num_atoms.png
analysis_outputs/mean_error_by_atom_count.png
analysis_outputs/error_by_has_N.png
analysis_outputs/error_by_has_O.png
analysis_outputs/error_by_has_F.png
analysis_outputs/test_error_analysis.csv
analysis_outputs/worst_20_predictions.csv
analysis_outputs/error_group_summary.csv
```

## Suggested report figures

Use these figures in the report:

1. `results/training_curves.png` for the capacity ablation.
2. `results/capacity_ablation_results.csv` as the table for layers / hidden dimensions / parameters / test MAE.
3. `analysis_outputs/latent_space_error_pca.png` for latent-space error distribution.
4. `analysis_outputs/latent_space_num_atoms_pca.png` for latent-space molecular-size structure.
5. `analysis_outputs/error_vs_num_atoms.png` or `analysis_outputs/mean_error_by_atom_count.png` for failure-case analysis.
6. `analysis_outputs/worst_20_predictions.csv` for the largest-error molecules.

## Report wording template

### Contribution 1: Latent-space analysis

We extracted the graph-level molecular embeddings before the final regression layer of the trained GCN model. These embeddings were projected to two dimensions using PCA and colored by absolute prediction error and molecule size. This analysis allows us to examine whether the learned representation organizes molecules according to structural complexity and whether high-error molecules occupy specific regions of latent space.

### Contribution 2: Capacity ablation

We varied the capacity of the GCN while keeping the optimizer, learning rate, batch size, scheduler, training epochs, and data split fixed. This isolates the effect of model size. A larger model can represent more complex molecular patterns, but may also overfit or suffer from over-smoothing on small molecular graphs.

### Contribution 3: Error analysis

We analyzed the best checkpoint at the per-molecule level instead of only reporting aggregate MAE. Molecules were grouped by number of atoms and atom composition. This reveals whether the model makes systematic errors on larger molecules or molecules containing specific atom types such as N, O, or F.
