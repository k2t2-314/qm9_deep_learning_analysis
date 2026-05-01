from torch_geometric.datasets import QM9
import torch

dataset = QM9(root='./data/QM9')

print(f"Number of molecules: {len(dataset)}")
print(f"Number of features per atom: {dataset.num_features}")
print(f"Number of targets: {dataset.num_classes}")

# Look at one sample
mol = dataset[0]
print(f"\nSample molecule:")
print(f"  Atom features (x): {mol.x.shape}")
print(f"  Edge index:        {mol.edge_index.shape}")
print(f"  Edge features:     {mol.edge_attr.shape}")
print(f"  3D coordinates:    {mol.pos.shape}")
print(f"  Targets (y):       {mol.y.shape}")

target_names = [
    'mu', 'alpha', 'homo', 'lumo', 'gap',
    'r2', 'zpve', 'U0', 'U', 'H',
    'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom',
    'G_atom', 'A', 'B', 'C'
]

# Stats for all 19 targets
print(f"\nStats for all targets across dataset:")
print(f"{'Index':<6} {'Name':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 56)
for i, name in enumerate(target_names):
    vals = dataset._data.y[:, i]
    print(f"  [{i}]  {name:<10} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.min():>10.4f} {vals.max():>10.4f}")