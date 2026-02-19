import torch
import torch.nn as nn
import torch.nn.functional as F
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
import mrcfile
import argparse
from loss import HybridDiceLoss

# --- IMPORT CUSTOM MODULES ---
sys.path.append(str(Path(__file__).resolve().parent))
from architecture import SCUNet               
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
    "epochs": 60,          # Reduced from 100 (Model learns fast)
    "lr": 1e-4,
    "ligand_dim": 1024,
    "voxel_size": 0.5,
    "dice_weight": 0.7,    # High priority on shape
    "mse_weight": 0.3,     # Lower priority on exact intensity
    "augmentation": False  # Keep False for now as discussed
}

# --- HELPER CLASS FOR MRC SAVING ---
class SimpleCell:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


# --- DATASET CLASS ---
class CryoEMDataset(Dataset):
    def __init__(self, h5_path, ligand_dim=1024, augment=False):
        self.h5_path = h5_path
        self.ligand_dim = ligand_dim
        self.augment = augment 
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
        
        # We don't strictly need the mask for Dice, but we pass it for consistency
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

# --- HELPER: LOCAL PLOTTING ---
def plot_metrics_local(history, save_path):
    if len(history['train_loss']) < 1: return
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].plot(epochs, history['train_loss'], label='Train')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val')
    axs[0, 0].set_title('Total Loss (Dice + MSE)')
    axs[0, 0].legend()
    
    axs[0, 1].plot(epochs, history['masked_mse'], 'r')
    axs[0, 1].set_title('Raw MSE (Monitoring Only)')
    axs[0, 1].set_yscale('log')
    
    axs[1, 0].plot(epochs, history['grad_norm'], 'g')
    axs[1, 0].set_title('Gradient Norm')
    
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
    Saves predictions using the EXACT conventions from the generation script.
    Filename format: pdbid_ligandname_type.mrc
    """
    print(f"   >>> Generating {num_samples} MRC samples for inspection...")
    model.eval()
    output_dir = RESULTS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(num_samples, len(subset))
    if n_samples == 0: return
    
    # Randomly select indices from the validation subset
    indices = random.sample(subset.indices, n_samples)
    
    with h5py.File(HDF5_FILE, 'r') as f:
        for idx in indices:
            # 1. READ METADATA EXACTLY AS GENERATED
            pdb_id = f['pdb_ids'][idx].decode('utf-8')
            lig_name = f['ligand_names'][idx].decode('utf-8') # e.g. "ATP_1"
            
            exp_density = f['exp_density'][idx]
            protein_mask = f['maps'][idx][0]
            ground_truth = f['ground_truth_maps'][idx]
            phys_origin = f['physical_origin'][idx] # The exact origin saved during generation
            smiles = f['ligand_smiles'][idx].decode('utf-8')
            
            # 2. CONSTRUCT BASE NAME
            # This ensures we know exactly which PDB and which Ligand instance this is
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
            
            # 3. CALCULATE CELL DIMENSIONS (EXACTLY AS IN GENERATION)
            # Generation used: GRID_SIZE * TARGET_VOXEL_SIZE
            data_to_save = pred_np.T
            nx, ny, nz = data_to_save.shape
            
            # We use the config voxel size which matches generation (0.5)
            total_a = nx * CONFIG['voxel_size']
            total_b = ny * CONFIG['voxel_size']
            total_c = nz * CONFIG['voxel_size']
            mock_cell = SimpleCell(total_a, total_b, total_c)
            
            # 4. SAVE (Prediction, Ground Truth, Input)
            print(f"      Saving: {base_name}_pred.mrc")
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
    parser.set_defaults(augment=False) # Default to False based on analysis
    args = parser.parse_args()
    
    CONFIG["augmentation"] = args.augment

    # --- SETUP PATHS ---
    aug_tag = "aug" if CONFIG["augmentation"] else "no_aug"
    RUN_NAME = f"run_{TIMESTAMP}_DICE_LOSS_{aug_tag}" # Tagged with DICE
    RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    wandb.init(project="cryoem-ligand-fitting", name=RUN_NAME, config=CONFIG, save_code=True)
    print(f"--- Training {RUN_NAME} on {DEVICE} ---")
    print(f"--- FUNDAMENTAL CHANGE: Using Hybrid Dice ({CONFIG['dice_weight']}) + MSE ({CONFIG['mse_weight']}) Loss ---")

    # --- DATASET ---
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
    model = SCUNet(in_nc=2, ligand_dim=CONFIG["ligand_dim"], window_size=4).to(DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # *** CHANGE 1: NEW LOSS ***
    criterion = HybridDiceLoss(dice_weight=CONFIG['dice_weight'], mse_weight=CONFIG['mse_weight']).to(DEVICE)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norm': [], 'masked_mse': []}
    best_val_loss = float('inf') 
    
    try:
        for epoch in range(CONFIG["epochs"]):
            # --- TRAIN ---
            model.train()
            train_loss_accum = 0
            grad_norm_accum = 0
            
            for inputs, lig_emb, targets, lig_masks in train_loader:
                inputs, lig_emb, targets, lig_masks = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE), lig_masks.to(DEVICE)
                optimizer.zero_grad()
                preds = model(inputs, lig_emb)
                
                # New Loss
                loss = criterion(preds, targets, mask=lig_masks) 
                loss.backward()
                
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None: total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm_accum += (total_norm ** 0.5)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_accum += loss.item()

            # --- VALIDATE ---
            model.eval()
            val_loss_accum = 0
            masked_mse_accum = 0
            with torch.no_grad():
                for inputs, lig_emb, targets, lig_masks in val_loader:
                    inputs, lig_emb, targets, lig_masks = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE), lig_masks.to(DEVICE)
                    preds = model(inputs, lig_emb)
                    
                    # Val Loss (Dice+MSE)
                    val_loss_accum += criterion(preds, targets, mask=lig_masks).item()
                    
                    # Monitor Raw MSE specifically on the ligand (for legacy comparison)
                    roi_mask = (lig_masks > 0.5)
                    if roi_mask.sum() > 0:
                        masked_mse_accum += ((preds - targets)**2)[roi_mask].mean().item()

            # --- METRICS ---
            avg_train_loss = train_loss_accum / len(train_loader)
            avg_val_loss = val_loss_accum / len(val_loader)
            avg_masked_mse = masked_mse_accum / len(val_loader)
            avg_grad_norm = grad_norm_accum / len(train_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['masked_mse'].append(avg_masked_mse)
            history['grad_norm'].append(avg_grad_norm)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            scheduler.step(avg_val_loss)
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "masked_mse": avg_masked_mse, "epoch": epoch})
            
            print(f"Epoch {epoch+1:03d} | Val Loss (Dice+MSE): {avg_val_loss:.5f} | Raw Ligand MSE: {avg_masked_mse:.5f}")
            plot_metrics_local(history, RESULTS_DIR / "training_metrics.png")
            
            # --- CHECKPOINTING ---
            # IMPORTANT: We now save based on the Validation Loss (Dice+MSE), not just MSE
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"   >>> New Best Val Loss! ({best_val_loss:.5f}) Saving weights...")
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
            if isinstance(model, nn.DataParallel):
                model.load_state_dict(state_dict)
            else:
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            
            # *** GENERATE SAMPLES WITH EXACT NAMING ***
            save_mrc_samples(model, val_data, "final_best_predictions", DEVICE, num_samples=5)
        else:
            print("Warning: No best model found.")

        plot_metrics_local(history, RESULTS_DIR / "final_training_metrics.png")
        wandb.finish()
        print(f"--- Done. Check results in {RESULTS_DIR} ---")

if __name__ == "__main__":
    main()