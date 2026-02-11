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
import wandb  # <--- THE MAGIC TOOL

# --- IMPORT CUSTOM MODULES ---
sys.path.append(str(Path(__file__).resolve().parent))
from architecture import SCUNet               
from loss import EMReadyLikeLoss              
from utils_common import save_mrc_with_origin

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Generate a Timestamp for local file storage (still useful for backups)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"run_{TIMESTAMP}"
RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HDF5_FILE = DATA_DIR / "ml_dataset_FINAL.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HYPERPARAMETERS ---
CONFIG = {
    "batch_size": 16,
    "epochs": 100,
    "lr": 2e-4,
    "ligand_dim": 1024,
    "voxel_size": 0.5,
    "ligand_weight": 500.0,
    "ssim_weight": 0.2,
    "architecture": "SCUNet",
    "dataset_version": "FINAL"
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

        exp_density = self.h5_file['exp_density'][idx]
        protein_mask = self.h5_file['maps'][idx][0]
        
        # Input: [Density, Protein_Mask]
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

# --- HELPER: LOG IMAGES TO WANDB ---
def log_visuals_to_wandb(model, loader, device, epoch):
    """
    Takes one batch from validation, predicts, and logs the central slice 
    of the first 3 examples to WandB.
    """
    model.eval()
    inputs, lig_emb, targets, _ = next(iter(loader))
    
    # Pick first 3 examples
    n_show = min(3, inputs.shape[0])
    inputs = inputs[:n_show].to(device)
    lig_emb = lig_emb[:n_show].to(device)
    targets = targets[:n_show]
    
    with torch.no_grad():
        preds = model(inputs, lig_emb).cpu()
        
    images = []
    for i in range(n_show):
        # Get central slice index
        d = inputs.shape[2] // 2
        
        # Input Density (Channel 0)
        img_in = inputs[i, 0, :, :, d].cpu().numpy()
        # Prediction (Channel 0)
        img_pred = preds[i, 0, :, :, d].numpy()
        # Ground Truth (Channel 0)
        img_gt = targets[i, 0, :, :, d].numpy()
        
        # Combine side-by-side: Input | Pred | GT
        combined = np.concatenate([img_in, img_pred, img_gt], axis=1)
        
        # Normalize for display (0-1)
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        
        caption = f"Epoch {epoch} | Sample {i} (In / Pred / GT)"
        images.append(wandb.Image(combined, caption=caption))
        
    wandb.log({"Validation Visuals": images}, commit=False)

# --- MAIN LOOP ---
def main():
    # 1. INITIALIZE WANDB
    wandb.init(
        project="cryoem-ligand-fitting",
        name=RUN_NAME,
        config=CONFIG,
        dir=str(PROJECT_ROOT) # Store metadata in project root
    )
    
    print(f"--- Initializing Training {RUN_NAME} on {DEVICE} ---")
    
    if not HDF5_FILE.exists(): raise FileNotFoundError(f"Dataset not found at {HDF5_FILE}")
        
    full_dataset = CryoEMDataset(HDF5_FILE, ligand_dim=CONFIG["ligand_dim"])
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8, pin_memory=True)
    
    model = SCUNet(in_nc=2, ligand_dim=CONFIG["ligand_dim"], window_size=4).to(DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
    # Watch the model (logs gradients and topology automatically)
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion = EMReadyLikeLoss(
        ligand_weight=CONFIG["ligand_weight"], 
        ssim_weight=CONFIG["ssim_weight"]
    ).to(DEVICE)
    
    best_loss = float('inf')
    
    try:
        for epoch in range(CONFIG["epochs"]):
            model.train()
            train_loss_accum = 0
            grad_norm_accum = 0
            
            # --- TRAINING ---
            for i, (inputs, lig_emb, targets, lig_masks) in enumerate(train_loader):
                inputs, lig_emb, targets, lig_masks = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE), lig_masks.to(DEVICE)
                
                optimizer.zero_grad()
                preds = model(inputs, lig_emb)
                loss = criterion(preds, targets, lig_masks) 
                loss.backward()
                
                # Calculate Grad Norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5
                grad_norm_accum += grad_norm
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_accum += loss.item()
                
                # Log batch-level metrics (every 10 batches)
                if i % 10 == 0:
                    wandb.log({
                        "batch_train_loss": loss.item(),
                        "batch_grad_norm": grad_norm,
                        "lr": optimizer.param_groups[0]['lr']
                    })

            # --- VALIDATION ---
            model.eval()
            val_loss_accum = 0
            masked_mse_accum = 0
            
            with torch.no_grad():
                for inputs, lig_emb, targets, lig_masks in val_loader:
                    inputs, lig_emb, targets, lig_masks = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE), lig_masks.to(DEVICE)
                    preds = model(inputs, lig_emb)
                    
                    val_loss_accum += criterion(preds, targets, lig_masks).item()
                    
                    # MSE only inside the ligand mask
                    roi_mask = (lig_masks > 0.5)
                    if roi_mask.sum() > 0:
                        masked_mse = ((preds - targets)**2)[roi_mask].mean().item()
                    else:
                        masked_mse = 0.0
                    masked_mse_accum += masked_mse

            avg_train_loss = train_loss_accum / len(train_loader)
            avg_val_loss = val_loss_accum / len(val_loader)
            avg_masked_mse = masked_mse_accum / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LigMSE: {avg_masked_mse:.5f}")
            
            # --- WANDB LOGGING (EPOCH LEVEL) ---
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_masked_mse": avg_masked_mse,
                "avg_grad_norm": grad_norm_accum / len(train_loader)
            }
            
            # Log images every 5 epochs
            if (epoch + 1) % 5 == 0:
                log_visuals_to_wandb(model, val_loader, DEVICE, epoch + 1)
            
            wandb.log(log_dict)
            
            # --- SAVE CHECKPOINT ---
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), RESULTS_DIR / "best_model.pth")
                # Also save to wandb cloud
                wandb.save(str(RESULTS_DIR / "best_model.pth"))

    except KeyboardInterrupt:
        print("\nStopped manually.")

    # --- FINISH ---
    print(f"\n--- Run Complete. Best Loss: {best_loss:.5f} ---")
    wandb.finish()

if __name__ == "__main__":
    main()