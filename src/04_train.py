import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import h5py
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Custom Imports
from architecture import SCUNet, CustomLoss

# --- CONFIG ---
HDF5_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "ml_dataset.h5"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-4

# Ensure directories exist
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)


# ----------------------------
# HELPER: METRICS & PLOTTING
# ----------------------------
def get_grad_norm(model):
    """Calculates the global norm of gradients to monitor stability."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def plot_metrics(history, save_path):
    """Plots Loss, LR, Gradient Norm, and SSIM."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Loss
    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss Curves (MSE + SSIM)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Learning Rate
    axs[0, 1].plot(epochs, history['lr'], color='orange')
    axs[0, 1].set_title('Learning Rate Schedule')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].grid(True)

    # 3. Gradient Norm
    axs[1, 0].plot(epochs, history['grad_norm'], color='green')
    axs[1, 0].set_title('Gradient Norm (Stability)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_yscale('log')  # Log scale highlights spikes
    axs[1, 0].grid(True)

    # 4. SSIM Metric
    axs[1, 1].plot(epochs, history['val_ssim'], color='purple')
    axs[1, 1].set_title('Validation SSIM (Structure Quality)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ----------------------------
# DATASET CLASS
# ----------------------------
class CryoEMDataset(Dataset):
    def __init__(self, h5_path, ligand_dim=1024):
        self.h5_path = h5_path
        self.ligand_dim = ligand_dim
        self.h5_file = None
        
        # Open once briefly to get length
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['pdb_ids'])

    def _get_fingerprint(self, smiles_bytes):
        """Converts SMILES bytes to Morgan Fingerprint Tensor."""
        try:
            smiles = smiles_bytes.decode('utf-8')
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(self.ligand_dim, dtype=torch.float32)
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.ligand_dim)
            arr = np.array(fp, dtype=np.float32)
            return torch.from_numpy(arr)
        except Exception:
            return torch.zeros(self.ligand_dim, dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        # 1. Get Experimental Density (Shape: 96, 96, 96)
        exp_density = self.h5_file['exp_density'][idx]
        
        # 2. Get Protein Mask (Shape: 96, 96, 96) from 'maps' [channel 0]
        # Note: 'maps' is (N, 2, 96, 96, 96) -> [0]=Protein, [1]=LigandMask
        protein_mask = self.h5_file['maps'][idx][0]

        # 3. Create Input Tensor (2, 96, 96, 96)
        # Channel 0: Experimental Density
        # Channel 1: Protein Context
        input_tensor = np.stack([exp_density, protein_mask], axis=0)

        # 4. Get Ground Truth (Target)
        target = self.h5_file['ground_truth_maps'][idx]
        target = np.expand_dims(target, axis=0) # (1, 96, 96, 96)

        # 5. Get Ligand Embedding
        smiles = self.h5_file['ligand_smiles'][idx]
        ligand_emb = self._get_fingerprint(smiles)

        return (
            torch.from_numpy(input_tensor).float(), 
            ligand_emb.float(), 
            torch.from_numpy(target).float()
        )


# ----------------------------
# TRAINING LOOP
# ----------------------------
def main():
    print(f"--- Training on {DEVICE} ---")
    
    # Initialize Dataset
    if not HDF5_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {HDF5_FILE}")

    dataset = CryoEMDataset(HDF5_FILE)
    
    # Split Train/Val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    # DataLoaders (pin_memory=True speeds up transfer to CUDA)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model
    # in_nc=2 matches (ExpDensity + ProteinMask)
    model = SCUNet(in_nc=2, ligand_dim=1024).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # LR Scheduler: Reduces LR by half if validation loss stagnates for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Loss Function
    try:
        criterion = CustomLoss()
    except NameError:
        print("CustomLoss not found, using MSELoss.")
        criterion = nn.MSELoss()

    # Metrics History
    history = {
        'train_loss': [], 'val_loss': [], 
        'lr': [], 'grad_norm': [], 'val_ssim': []
    }

    best_loss = float('inf')

    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        # === TRAIN ===
        model.train()
        t_loss = 0
        total_grad_norm = 0
        
        for inputs, lig_emb, targets in train_loader:
            inputs, lig_emb, targets = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(inputs, lig_emb)
            loss = criterion(preds, targets)
            loss.backward()
            
            # Monitoring: Calculate Gradient Norm
            grad_norm = get_grad_norm(model)
            total_grad_norm += grad_norm
            
            # Stability: Clip Gradients (prevents explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            t_loss += loss.item()

        # === VALIDATION ===
        model.eval()
        v_loss = 0
        total_ssim = 0
        
        with torch.no_grad():
            for inputs, lig_emb, targets in val_loader:
                inputs, lig_emb, targets = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE)
                preds = model(inputs, lig_emb)
                
                # 1. Loss
                v_loss += criterion(preds, targets).item()
                
                # 2. SSIM (Approximate calculation for tracking)
                inputs_mean = preds.mean(dim=(2, 3, 4), keepdim=True)
                targets_mean = targets.mean(dim=(2, 3, 4), keepdim=True)
                cov = ((preds - inputs_mean) * (targets - targets_mean)).mean(dim=(2, 3, 4))
                inputs_var = preds.var(dim=(2, 3, 4))
                targets_var = targets.var(dim=(2, 3, 4))
                ssim_batch = ((2 * cov + 1e-6) / (inputs_var + targets_var + 1e-6)).mean().item()
                total_ssim += ssim_batch

        # === AVERAGES & UPDATES ===
        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader)
        avg_grad = total_grad_norm / len(train_loader)
        avg_ssim = total_ssim / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step Scheduler
        scheduler.step(avg_v)
        
        # Update History
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        history['lr'].append(current_lr)
        history['grad_norm'].append(avg_grad)
        history['val_ssim'].append(avg_ssim)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {avg_t:.5f} | Val: {avg_v:.5f} | SSIM: {avg_ssim:.4f} | Grad: {avg_grad:.2f} | LR: {current_lr:.1e}")
        
        # Plot every epoch
        plot_metrics(history, save_path=PLOT_DIR / "training_metrics.png")

        # Save Best Model
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  --> Best Model Saved (Val Loss: {best_loss:.5f})")

        # Save Checkpoint (e.g., every 10 epochs or just "latest")
        torch.save(model.state_dict(), CHECKPOINT_DIR / "latest_checkpoint.pth")

if __name__ == "__main__":
    main()