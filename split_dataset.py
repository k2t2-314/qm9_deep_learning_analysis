import torch
import numpy as np
import os

SEED      = 42
N_TOTAL   = 130831
N_TRAIN   = 110831
N_VAL     = 10000
N_TEST    = 10000

torch.manual_seed(SEED)
perm = torch.randperm(N_TOTAL).numpy()

train_idx = perm[:N_TRAIN]
val_idx   = perm[N_TRAIN:N_TRAIN + N_VAL]
test_idx  = perm[N_TRAIN + N_VAL:]

os.makedirs('data', exist_ok=True)
np.savez('data/split_42.npz',
         train_idx=train_idx,
         val_idx=val_idx,
         test_idx=test_idx)

print(f"Train: {len(train_idx)}")
print(f"Val:   {len(val_idx)}")
print(f"Test:  {len(test_idx)}")
print("Saved to data/split_42.npz")