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

# --- IMPORT CUSTOM MODULES ---
sys.path.append(str(Path(__file__).resolve().parent))
from architecture import SCUNet               
from loss import EMReadyLikeLoss              # <--- CHANGED IMPORT
from utils_common import save_mrc_with_origin

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOT_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"

HDF5_FILE = DATA_DIR / "ml_dataset.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HYPERPARAMETERS ---
BATCH_SIZE = 8
EPOCHS = 100
LR = 2e-4             # Good starting point for Smooth L1
LIGAND_DIM = 1024
VOXEL_SIZE = 0.5 

# --- LOSS CONFIGURATION ---
LIGAND_WEIGHT = 500.0  # High penalty for missing the ligand
SSIM_WEIGHT = 0.2      # Small weight for SSIM to start (Paper uses sum, but we balance it)

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

        exp_density = self.h5_file['exp_density'][idx]
        protein_mask = self.h5_file['maps'][idx][0]
        
        input_tensor = np.stack([exp_density, protein_mask], axis=0)
        
        target = self.h5_file['ground_truth_maps'][idx]
        target = np.expand_dims(target, axis=0)

        # LOAD THE MASK (0 or 1)
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

# --- HELPER: PLOTTING ---
def plot_metrics(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].plot(epochs, history['train_loss'], label='Train')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val')
    axs[0, 0].set_title(f'Total Loss (SmoothL1 + SSIM)')
    axs[0, 0].legend()
    axs[0, 0].set_yscale('log')
    
    axs[0, 1].plot(epochs, history['masked_mse'], 'r')
    axs[0, 1].set_title('Ligand Only MSE (Raw Error)')
    axs[0, 1].set_yscale('log')
    
    axs[1, 0].plot(epochs, history['grad_norm'], 'g')
    axs[1, 0].set_title('Gradient Norm')
    
    axs[1, 1].plot(epochs, history['lr'], 'orange')
    axs[1, 1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- HELPER: 3D EXPORT ---
def save_mrc_samples(model, subset, folder_name, device, num_samples=5):
    model.eval()
    output_dir = RESULTS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(num_samples, len(subset))
    if n_samples == 0: return
    indices = random.sample(subset.indices, n_samples)
    
    with h5py.File(HDF5_FILE, 'r') as f:
        print(f"--- Exporting {len(indices)} samples to {output_dir} ---")
        for idx in indices:
            exp_density = f['exp_density'][idx]
            protein_mask = f['maps'][idx][0]
            ground_truth = f['ground_truth_maps'][idx]
            phys_origin = f['physical_origin'][idx]
            pdb_id = f['pdb_ids'][idx].decode('utf-8')
            smiles = f['ligand_smiles'][idx].decode('utf-8')
            
            input_np = np.stack([exp_density, protein_mask], axis=0)
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).float().to(device)
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=LIGAND_DIM)
                lig_emb = torch.from_numpy(np.array(fp, dtype=np.float32)).unsqueeze(0).float().to(device)
            except:
                lig_emb = torch.zeros(1, LIGAND_DIM).float().to(device)

            with torch.no_grad():
                pred_tensor = model(input_tensor, lig_emb)
            
            pred_np = pred_tensor.cpu().numpy().squeeze()
            base_name = f"{pdb_id}"
            
            save_mrc_with_origin(pred_np.T, output_dir / f"{base_name}_pred.mrc", VOXEL_SIZE, phys_origin)
            save_mrc_with_origin(ground_truth.T, output_dir / f"{base_name}_gt.mrc", VOXEL_SIZE, phys_origin)
            save_mrc_with_origin(exp_density.T, output_dir / f"{base_name}_input.mrc", VOXEL_SIZE, phys_origin)
            print(f"   Saved {base_name}")

# --- MAIN LOOP ---
def main():
    print(f"--- Initializing Training on {DEVICE} ---")
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if not HDF5_FILE.exists(): raise FileNotFoundError(f"Dataset not found at {HDF5_FILE}")
        
    full_dataset = CryoEMDataset(HDF5_FILE, ligand_dim=LIGAND_DIM)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = SCUNet(in_nc=2, ligand_dim=LIGAND_DIM).to(DEVICE)
    if gpu_count > 1: model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # --- EMREADY-STYLE LOSS ---
    criterion = EMReadyLikeLoss(
        ligand_weight=LIGAND_WEIGHT, 
        ssim_weight=SSIM_WEIGHT
    ).to(DEVICE)
    
    print(f"Loss Config: SmoothL1 (Weight={LIGAND_WEIGHT}) + SSIM (Weight={SSIM_WEIGHT})")

    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norm': [], 'masked_mse': []}
    best_loss = float('inf')
    
    print("\n--- Starting Epochs ---")
    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss_accum = 0
            grad_norm_accum = 0
            
            for inputs, lig_emb, targets, lig_masks in train_loader:
                inputs = inputs.to(DEVICE)
                lig_emb = lig_emb.to(DEVICE)
                targets = targets.to(DEVICE)
                lig_masks = lig_masks.to(DEVICE)
                
                optimizer.zero_grad()
                preds = model(inputs, lig_emb)
                
                # Forward pass through new loss
                loss = criterion(preds, targets, lig_masks) 
                loss.backward()
                
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
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
                    
                    # Monitor: Pure MSE inside the mask
                    roi_mask = (lig_masks > 0.5)
                    if roi_mask.sum() > 0:
                        masked_mse = ((preds - targets)**2)[roi_mask].mean().item()
                    else:
                        masked_mse = 0.0
                    masked_mse_accum += masked_mse

            avg_train_loss = train_loss_accum / len(train_loader)
            avg_val_loss = val_loss_accum / len(val_loader)
            avg_masked_mse = masked_mse_accum / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['grad_norm'].append(grad_norm_accum / len(train_loader))
            history['masked_mse'].append(avg_masked_mse)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | Ligand MSE: {avg_masked_mse:.5f}")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, CHECKPOINT_DIR / "best_model.pth")
            
            if (epoch + 1) % 10 == 0:
                plot_metrics(history, PLOT_DIR / "training_metrics.png")

    except KeyboardInterrupt:
        print("\nStopped manually.")

    # --- EXPORT ---
    print("\n--- Exporting Verification Samples ---")
    best_path = CHECKPOINT_DIR / "best_model.pth"
    if best_path.exists():
        state_dict = torch.load(best_path, map_location=DEVICE)
        if isinstance(model, nn.DataParallel): model.module.load_state_dict(state_dict)
        else: model.load_state_dict(state_dict)
    
    save_mrc_samples(model, train_data, "predictions_train", DEVICE)
    save_mrc_samples(model, val_data, "predictions_val", DEVICE)
    print(f"Done. Check {RESULTS_DIR}/predictions_val")

if __name__ == "__main__":
    main()