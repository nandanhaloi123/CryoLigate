import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
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
import argparse

# --- IMPORT CUSTOM MODULES ---
# Ensure 'architecture.py', 'loss.py', and 'utils_common.py' are in the same folder
sys.path.append(str(Path(__file__).resolve().parent))
from architecture import SCUNet               
from loss import HybridROILoss  # <--- Importing your new separate loss file
from utils_common import save_mrc_with_origin

# --- CONFIGURATION DEFAULTS ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# 1. SETUP DIRS
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HDF5_FILE = DATA_DIR / "ml_dataset_FINAL.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "batch_size": 16,
    "epochs": 60,
    "lr": 1e-4,
    "ligand_dim": 768,
    "voxel_size": 0.5,
    
    # --- FUNDAMENTAL CHANGE: LOSS WEIGHTS ---
    # We want the model to scream if it misses the ligand (Masked MSE),
    # while gently keeping the background clean (Global MSE).
    "dice_weight": 0.5,     # Encourages shape overlap
    "masked_weight": 10.0,  # CRITICAL: 10x penalty for errors inside the ligand
    "global_weight": 0.1,   # Low weight because background is easy
    "ssim_weight": 0.0,     # Start at 0.0. Turn up to 0.1 later if texture is blurry.
    
    "augmentation": False 
}

# --- HELPER CLASS FOR MRC SAVING ---
class SimpleCell:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

# --- DATASET CLASS ---
# class CryoEMDataset(Dataset):
#     def __init__(self, h5_path, ligand_dim=1024, augment=False):
#         self.h5_path = h5_path
#         self.ligand_dim = ligand_dim
#         self.augment = augment 
#         self.h5_file = None
        
#         with h5py.File(h5_path, 'r') as f:
#             self.length = len(f['pdb_ids'])

#     def _get_fingerprint(self, smiles_bytes):
#         try:
#             smiles = smiles_bytes.decode('utf-8')
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None: return torch.zeros(self.ligand_dim, dtype=torch.float32)
#             fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.ligand_dim)
#             return torch.from_numpy(np.array(fp, dtype=np.float32))
#         except:
#             return torch.zeros(self.ligand_dim, dtype=torch.float32)
    
#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         if self.h5_file is None:
#             self.h5_file = h5py.File(self.h5_path, 'r')
        
#         exp_density = self.h5_file['exp_density'][idx]
#         protein_mask = self.h5_file['maps'][idx][0]
#         input_tensor = np.stack([exp_density, protein_mask], axis=0)
        
#         target = self.h5_file['ground_truth_maps'][idx]
#         target = np.expand_dims(target, axis=0)
        
#         lig_mask = self.h5_file['masks'][idx]
#         lig_mask = np.expand_dims(lig_mask, axis=0)

#         smiles = self.h5_file['ligand_smiles'][idx]
#         ligand_emb = self._get_fingerprint(smiles)

#         return (
#             torch.from_numpy(input_tensor).float(), 
#             ligand_emb.float(), 
#             torch.from_numpy(target).float(),
#             torch.from_numpy(lig_mask).float() 
#         )

class CryoEMDataset(Dataset):
    def __init__(self, h5_path, embeddings_path, augment=False):
        self.h5_path = h5_path
        self.augment = augment 
        self.h5_file = None
        
        # Load the pre-computed MolFormer embeddings
        # Expected shape: [N_samples, 768]
        print(f"Loading embeddings from {embeddings_path}...")
        self.ligand_embeddings = torch.load(embeddings_path)
        
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['pdb_ids'])

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
        
        lig_mask = self.h5_file['masks'][idx]
        lig_mask = np.expand_dims(lig_mask, axis=0)

        # Look up the pre-computed embedding by index
        # This is incredibly fast compared to running the model
        ligand_emb = self.ligand_embeddings[idx]

        return (
            torch.from_numpy(input_tensor).float(), 
            ligand_emb.float(), 
            torch.from_numpy(target).float(),
            torch.from_numpy(lig_mask).float() 
        )
        
# --- HELPER: LOCAL PLOTTING ---
def plot_metrics_local(history, save_path):
    if len(history['train_loss']) < 1: return
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].plot(epochs, history['train_loss'], label='Train')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val')
    axs[0, 0].set_title('Total Weighted Loss')
    axs[0, 0].legend()
    
    # This is the most important graph: Is the ligand error dropping?
    axs[0, 1].plot(epochs, history['masked_mse'], 'r')
    axs[0, 1].set_title('Masked MSE (Ligand Quality)')
    axs[0, 1].set_yscale('log')
    
    axs[1, 0].plot(epochs, history['dice_score'], 'purple')
    axs[1, 0].set_title('Dice Score (Shape Overlap)')
    
    axs[1, 1].plot(epochs, history['lr'], 'orange')
    axs[1, 1].set_title('Learning Rate')
    
    plt.tight_layout()
    try:
        plt.savefig(save_path)
    except: pass
    plt.close()

# --- HELPER: SAVE MRC PREDICTIONS (EXACT NAMING) ---
def save_mrc_samples(model, subset, folder_name, device, num_samples=5):
    """
    Saves predictions for visual inspection.
    Naming Convention: pdbid_ligandname_type.mrc (Matches generation script)
    """
    print(f"   >>> Generating {num_samples} MRC samples for inspection...")
    model.eval()
    output_dir = RESULTS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(num_samples, len(subset))
    if n_samples == 0: return
    
    indices = random.sample(subset.indices, n_samples)
    
    with h5py.File(HDF5_FILE, 'r') as f:
        for idx in indices:
            pdb_id = f['pdb_ids'][idx].decode('utf-8')
            lig_name = f['ligand_names'][idx].decode('utf-8') # e.g. "ATP_1"
            
            exp_density = f['exp_density'][idx]
            protein_mask = f['maps'][idx][0]
            ground_truth = f['ground_truth_maps'][idx]
            phys_origin = f['physical_origin'][idx]
            smiles = f['ligand_smiles'][idx].decode('utf-8')
            
            # Construct exact base name
            base_name = f"{pdb_id}_{lig_name}" 
            
            # Prepare Input
            input_np = np.stack([exp_density, protein_mask], axis=0)
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).float().to(device)
            
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
            
            # Create Mock Unit Cell for Header (Matches generation logic)
            data_to_save = pred_np.T
            nx, ny, nz = data_to_save.shape
            total_a = nx * CONFIG['voxel_size']
            total_b = ny * CONFIG['voxel_size']
            total_c = nz * CONFIG['voxel_size']
            mock_cell = SimpleCell(total_a, total_b, total_c)
            
            # Save files
            save_mrc_with_origin(data_to_save, output_dir / f"{base_name}_pred.mrc", mock_cell, phys_origin)
            save_mrc_with_origin(ground_truth.T, output_dir / f"{base_name}_gt.mrc", mock_cell, phys_origin)
            save_mrc_with_origin(exp_density.T, output_dir / f"{base_name}_input.mrc", mock_cell, phys_origin)

# --- MAIN LOOP ---
def main():
    global RUN_NAME, RESULTS_DIR
    
    # --- PARSE ARGS ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="Enable random rotation")
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.set_defaults(augment=False) # Default False for now
    args = parser.parse_args()
    
    CONFIG["augmentation"] = args.augment

    # --- SETUP PATHS ---
    aug_tag = "aug" if CONFIG["augmentation"] else "no_aug"
    RUN_NAME = f"run_{TIMESTAMP}_ROI_FUNDAMENTAL_{aug_tag}"
    RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    wandb.init(project="cryoem-ligand-fitting", name=RUN_NAME, config=CONFIG, save_code=True)
    print(f"--- Training {RUN_NAME} on {DEVICE} ---")
    print(f"--- Loss Config: Dice={CONFIG['dice_weight']}, ROI_MSE={CONFIG['masked_weight']}, Global={CONFIG['global_weight']} ---")

    # --- DATASET ---
    print("Loading dataset...")
    temp_ds = CryoEMDataset(HDF5_FILE)
    total_len = len(temp_ds)
    del temp_ds
    
    indices = list(range(total_len))
    split_point = int(0.9 * total_len)
    
    train_ds_full = CryoEMDataset(HDF5_FILE, ligand_dim=CONFIG["ligand_dim"], augment=CONFIG["augmentation"])
    train_data = Subset(train_ds_full, indices[:split_point])
    
    val_ds_full = CryoEMDataset(HDF5_FILE, ligand_dim=CONFIG["ligand_dim"], augment=False)
    val_data = Subset(val_ds_full, indices[split_point:])
    
    train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8)
    
    # --- MODEL ---
    print("Initializing SCUNet...")
    model = SCUNet(in_nc=2, ligand_dim=CONFIG["ligand_dim"], window_size=4).to(DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # --- LOSS FUNCTION INITIALIZATION ---
    criterion = HybridROILoss(
        dice_weight=CONFIG['dice_weight'], 
        masked_weight=CONFIG['masked_weight'], 
        global_weight=CONFIG['global_weight'],
        ssim_weight=CONFIG['ssim_weight']
    ).to(DEVICE)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'masked_mse': [], 'dice_score': []}
    best_masked_mse = float('inf') 
    
    try:
        for epoch in range(CONFIG["epochs"]):
            # --- TRAIN ---
            model.train()
            train_loss_accum = 0
            
            for inputs, lig_emb, targets, lig_masks in train_loader:
                inputs = inputs.to(DEVICE)
                lig_emb = lig_emb.to(DEVICE)
                targets = targets.to(DEVICE)
                lig_masks = lig_masks.to(DEVICE)
                
                optimizer.zero_grad()
                preds = model(inputs, lig_emb)
                
                # Unpack the 4 components from HybridROILoss
                loss, _, _, _, _ = criterion(preds, targets, lig_masks) 
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_accum += loss.item()

            # --- VALIDATE ---
            model.eval()
            val_loss_accum = 0
            masked_mse_accum = 0
            dice_score_accum = 0
            
            with torch.no_grad():
                for inputs, lig_emb, targets, lig_masks in val_loader:
                    inputs = inputs.to(DEVICE)
                    lig_emb = lig_emb.to(DEVICE)
                    targets = targets.to(DEVICE)
                    lig_masks = lig_masks.to(DEVICE)
                    
                    preds = model(inputs, lig_emb)
                    
                    # Unpack components
                    loss, dice_loss, masked_loss, global_loss, _ = criterion(preds, targets, lig_masks)
                    
                    val_loss_accum += loss.item()
                    masked_mse_accum += masked_loss.item()
                    dice_score_accum += (1 - dice_loss.item())

            # --- METRICS ---
            avg_train_loss = train_loss_accum / len(train_loader)
            avg_val_loss = val_loss_accum / len(val_loader)
            avg_masked_mse = masked_mse_accum / len(val_loader)
            avg_dice = dice_score_accum / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['masked_mse'].append(avg_masked_mse)
            history['dice_score'].append(avg_dice)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            scheduler.step(avg_val_loss)
            
            wandb.log({
                "train_loss": avg_train_loss, 
                "val_loss": avg_val_loss, 
                "masked_mse": avg_masked_mse, 
                "dice": avg_dice, 
                "epoch": epoch
            })
            
            print(f"Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.5f} | Ligand MSE: {avg_masked_mse:.5f} | Dice: {avg_dice:.3f}")
            plot_metrics_local(history, RESULTS_DIR / "training_metrics.png")
            
            # --- CHECKPOINTING ---
            # We save based on Ligand MSE because that is the hardest part of the problem.
            if avg_masked_mse < best_masked_mse:
                best_masked_mse = avg_masked_mse
                print(f"   >>> New Best Ligand MSE! ({best_masked_mse:.5f}) Saving weights...")
                torch.save(model.state_dict(), RESULTS_DIR / "best_model.pth")
                wandb.save(str(RESULTS_DIR / "best_model.pth"))

    except KeyboardInterrupt:
        print("\nStopped manually.")

    finally:
        print(f"\n--- Training Ended. Finalizing... ---")
        best_model_path = RESULTS_DIR / "best_model.pth"
        if best_model_path.exists():
            print(f"--- Loading Best Model Weights from {best_model_path} ---")
            state_dict = torch.load(best_model_path)
            
            # Handle DataParallel loading
            if isinstance(model, nn.DataParallel):
                model.load_state_dict(state_dict)
            else:
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            
            # Generate Final Samples
            save_mrc_samples(model, val_data, "final_best_predictions", DEVICE, num_samples=5)
        else:
            print("Warning: No best model found.")

        plot_metrics_local(history, RESULTS_DIR / "final_training_metrics.png")
        wandb.finish()
        print(f"--- Done. Check results in {RESULTS_DIR} ---")

if __name__ == "__main__":
    main()