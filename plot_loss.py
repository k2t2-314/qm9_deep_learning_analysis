import matplotlib.pyplot as plt
import csv

def load_metrics(path):
    train_mse, val_mae = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_mse.append(float(row['train_mse']))
            val_mae.append(float(row['val_mae']))
    return train_mse, val_mae

gcn_train, gcn_val           = load_metrics(r'D:\docs\qm9-project\logs\gcn\20260427_034205\metrics.csv')
gcn_dist_train, gcn_dist_val = load_metrics(r'D:\docs\qm9-project\logs\gcn_dist\20260429_155143\metrics.csv')
schnet_train, schnet_val     = load_metrics(r'D:\docs\qm9-project\logs\schnet\20260429_192357\metrics.csv')
dimenet_train, dimenet_val   = load_metrics(r'D:\docs\qm9-project\20260430_224522\20260430_224522\metrics.csv')

epochs = list(range(1, 151))

# Train MSE
plt.figure(figsize=(8, 5))
plt.plot(epochs, gcn_train,      label='GCN')
plt.plot(epochs, gcn_dist_train, label='GCN + Distance')
plt.plot(epochs, schnet_train,   label='SchNet')
plt.plot(epochs, dimenet_train,  label='DimeNet++')
plt.xlabel('Epoch')
plt.ylabel('Train MSE')
plt.title('Training Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logs/train_mse_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to logs/train_mse_curve.png")

# Val MAE
plt.figure(figsize=(8, 5))
plt.plot(epochs, gcn_val,      label='GCN')
plt.plot(epochs, gcn_dist_val, label='GCN + Distance')
plt.plot(epochs, schnet_val,   label='SchNet')
plt.plot(epochs, dimenet_val,  label='DimeNet++')
plt.xlabel('Epoch')
plt.ylabel('Val MAE (eV)')
plt.title('Validation MAE during Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('logs/val_mae_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to logs/val_mae_curve.png")