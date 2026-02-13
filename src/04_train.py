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
import random
import sys
from datetime import datetime
import wandb

# --- IMPORT CUSTOM MODULES ---
sys.path.append(str(Path(__file__).resolve().parent))
from architecture import SCUNet               
from loss import EMReadyLikeLoss              
from utils_common import save_mrc_with_origin

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 1. SETUP DIRS
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"run_{TIMESTAMP}"
RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HDF5_FILE = DATA_DIR / "ml_dataset_FINAL.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "batch_size": 16,
    "epochs": 100,
    "lr": 2e-4,
    "ligand_dim": 1024,
    "voxel_size": 0.5,
    "ligand_weight": 500.0,
    "ssim_weight": 0.2
}

# --- DATASET CLASS ---
class CryoEMDataset(Dataset):
    def __init__(self, h5_path, ligand_dim=1024):
        self.h5_path = h5_path
        self.ligand_dim = ligand_dim
        self.h5_file = None
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['pdb_ids'])

    def _get_fingerprint(self, smiles_bytes):
        try:
            smiles = smiles_bytes.decode('utf-8')
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return torch.zeros(self.ligand_dim, dtype=torch.float32)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.ligand_dim)
            return torch.from_numpy(np.array(fp, dtype=np.float32))
        except:
            return torch.zeros(self.ligand_dim, dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        # Load data
        exp_density = self.h5_file['exp_density'][idx]
        protein_mask = self.h5_file['maps'][idx][0]
        input_tensor = np.stack([exp_density, protein_mask], axis=0)
        
        target = self.h5_file['ground_truth_maps'][idx]
        target = np.expand_dims(target, axis=0)
        
        lig_mask = self.h5_file['masks'][idx]
        lig_mask = np.expand_dims(lig_mask, axis=0)

        smiles = self.h5_file['ligand_smiles'][idx]
        ligand_emb = self._get_fingerprint(smiles)

        return (
            torch.from_numpy(input_tensor).float(), 
            ligand_emb.float(), 
            torch.from_numpy(target).float(),
            torch.from_numpy(lig_mask).float() 
        )

# --- HELPER: LOCAL PLOTTING (Restored) ---
def plot_metrics_local(history, save_path):
    """Saves a PNG of the training curves locally."""
    if len(history['train_loss']) < 1: return
    
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Total Loss
    axs[0, 0].plot(epochs, history['train_loss'], label='Train')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val')
    axs[0, 0].set_title('Total Loss (Log Scale)')
    axs[0, 0].set_yscale('log')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.2)
    
    # 2. Ligand MSE (Pocket Error)
    axs[0, 1].plot(epochs, history['masked_mse'], 'r', label='Val Pocket MSE')
    axs[0, 1].set_title('Ligand Pocket MSE (Raw)')
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid(True, which="both", ls="-", alpha=0.2)
    
    # 3. Gradient Norm
    axs[1, 0].plot(epochs, history['grad_norm'], 'g')
    axs[1, 0].set_title('Gradient Norm (Stability)')
    axs[1, 0].grid(True)
    
    # 4. Learning Rate
    axs[1, 1].plot(epochs, history['lr'], 'orange')
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"   --> Updated plot: {save_path}")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")
    plt.close()


# --- HELPER CLASS FOR MRC SAVING ---
class SimpleCell:
    """
    A simple container to impersonate a UnitCell object.
    It holds the TOTAL physical size of the box (in Angstroms), not just one pixel.
    """
    def __init__(self, a, b, c):
        self.a = a  # Total length in X (Angstroms)
        self.b = b  # Total length in Y (Angstroms)
        self.c = c  # Total length in Z (Angstroms)
# --- HELPER: SAVE MRC PREDICTIONS ---

def save_mrc_samples(model, subset, folder_name, device, num_samples=3):
    """Saves .mrc files using a calculated UnitCell to keep utils_common happy."""
    model.eval()
    output_dir = RESULTS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pick indices (ensure we don't crash if dataset is smaller than num_samples)
    n_samples = min(num_samples, len(subset))
    if n_samples == 0: return
    indices = random.sample(subset.indices, n_samples)
    
    with h5py.File(HDF5_FILE, 'r') as f:
        for idx in indices:
            # Load Data
            exp_density = f['exp_density'][idx]
            protein_mask = f['maps'][idx][0]
            ground_truth = f['ground_truth_maps'][idx]
            phys_origin = f['physical_origin'][idx]
            pdb_id = f['pdb_ids'][idx].decode('utf-8')
            smiles = f['ligand_smiles'][idx].decode('utf-8')
            
            # Prepare Input
            input_np = np.stack([exp_density, protein_mask], axis=0)
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).float().to(device)
            
            # Handle Ligand Embedding
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CONFIG['ligand_dim'])
                lig_emb = torch.from_numpy(np.array(fp, dtype=np.float32)).unsqueeze(0).float().to(device)
            except:
                lig_emb = torch.zeros(1, CONFIG['ligand_dim']).float().to(device)

            # Predict
            with torch.no_grad():
                pred_tensor = model(input_tensor, lig_emb)
            
            pred_np = pred_tensor.cpu().numpy().squeeze()
            
            # --- THE FIX STARTS HERE ---
            # 1. Get the shape of the data we are saving
            # Note: We are saving the Transpose (.T), so we must swap dimensions for the calculation
            # pred_np is (Z, Y, X) -> pred_np.T is (X, Y, Z)
            data_to_save = pred_np.T
            nx, ny, nz = data_to_save.shape
            
            # 2. Calculate TOTAL box size in Angstroms (Shape * Voxel Size)
            # This is what utils_common expects as "unit_cell.a"
            total_a = nx * CONFIG['voxel_size']
            total_b = ny * CONFIG['voxel_size']
            total_c = nz * CONFIG['voxel_size']
            
            # 3. Create the object wrapper
            mock_cell = SimpleCell(total_a, total_b, total_c)
            
            # 4. Save using the mock cell
            base = f"{pdb_id}"
            save_mrc_with_origin(data_to_save, output_dir / f"{base}_pred.mrc", mock_cell, phys_origin)
            
            if not (output_dir / f"{base}_gt.mrc").exists():
                save_mrc_with_origin(ground_truth.T, output_dir / f"{base}_gt.mrc", mock_cell, phys_origin)
                save_mrc_with_origin(exp_density.T, output_dir / f"{base}_input.mrc", mock_cell, phys_origin)

# --- MAIN LOOP ---
def main():
    wandb.init(project="cryoem-ligand-fitting", name=RUN_NAME, config=CONFIG)
    print(f"--- Training {RUN_NAME} on {DEVICE} ---")
    print(f"--- Results saving to: {RESULTS_DIR} ---")
    
    full_dataset = CryoEMDataset(HDF5_FILE, ligand_dim=CONFIG["ligand_dim"])
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8)
    
    model = SCUNet(in_nc=2, ligand_dim=CONFIG["ligand_dim"], window_size=4).to(DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion = EMReadyLikeLoss(ligand_weight=CONFIG["ligand_weight"], ssim_weight=CONFIG["ssim_weight"]).to(DEVICE)
    
    # Local History Tracker
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norm': [], 'masked_mse': []}
    best_loss = float('inf')
    
    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()
            train_loss_accum = 0
            grad_norm_accum = 0
            
            for inputs, lig_emb, targets, lig_masks in train_loader:
                inputs, lig_emb, targets, lig_masks = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE), lig_masks.to(DEVICE)
                
                optimizer.zero_grad()
                preds = model(inputs, lig_emb)
                loss = criterion(preds, targets, lig_masks)
                loss.backward()
                
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None: total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm_accum += (total_norm ** 0.5)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_accum += loss.item()

            # --- VALIDATION ---
            model.eval()
            val_loss_accum = 0
            masked_mse_accum = 0
            
            with torch.no_grad():
                for inputs, lig_emb, targets, lig_masks in val_loader:
                    inputs, lig_emb, targets, lig_masks = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE), lig_masks.to(DEVICE)
                    preds = model(inputs, lig_emb)
                    val_loss_accum += criterion(preds, targets, lig_masks).item()
                    
                    roi_mask = (lig_masks > 0.5)
                    masked_mse = ((preds - targets)**2)[roi_mask].mean().item() if roi_mask.sum() > 0 else 0.0
                    masked_mse_accum += masked_mse

            # Metrics
            avg_train_loss = train_loss_accum / len(train_loader)
            avg_val_loss = val_loss_accum / len(val_loader)
            avg_masked_mse = masked_mse_accum / len(val_loader)
            avg_grad_norm = grad_norm_accum / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update History
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['grad_norm'].append(avg_grad_norm)
            history['masked_mse'].append(avg_masked_mse)
            history['lr'].append(current_lr)
            
            scheduler.step(avg_val_loss)
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "masked_mse": avg_masked_mse, "epoch": epoch})
            
            print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | PocketMSE: {avg_masked_mse:.5f}")
            
            # --- SAVE PLOTS & MODELS ---
            # 1. Save Plot Locally
            plot_metrics_local(history, RESULTS_DIR / "training_metrics.png")
            
            # 2. Checkpoint (If better)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                print("   >>> New Best Model! Saving checkpoint & predictions...")
                
                # Save Weights
                torch.save(model.state_dict(), RESULTS_DIR / "best_model.pth")
                wandb.save(str(RESULTS_DIR / "best_model.pth"))
                
                # Save Prediction Samples Immediately (So you see them even if run crashes)
                save_mrc_samples(model, val_data, "predictions_val_best", DEVICE)

    except KeyboardInterrupt:
        print("\nStopped manually.")

    finally:
        # Ensures this runs even if crushed
        print(f"\n--- Finalizing. Results in {RESULTS_DIR} ---")
        plot_metrics_local(history, RESULTS_DIR / "final_training_metrics.png")
        wandb.finish()

if __name__ == "__main__":
    main()