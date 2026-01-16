import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import h5py
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from pathlib import Path
import matplotlib.pyplot as plt
import os
import random
import mrcfile  # Required for exporting 3D files

# Custom Imports
from architecture import SCUNet, CustomLoss

# --- CONFIG ---
HDF5_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "ml_dataset.h5"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "mrc_visuals"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-4

# Ensure directories exist
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ----------------------------
# HELPER: METRICS & VISUALIZATION
# ----------------------------
def get_grad_norm(model):
    """Calculates the global norm of gradients to monitor stability."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def save_visual_check(model, dataset, epoch, device):
    """Saves a 2D slice visual comparison (Input vs Pred vs Target)."""
    model.eval()
    # Always pick sample 0 from the FULL dataset for consistency
    inputs, lig_emb, target = dataset[0]
    
    inputs = inputs.unsqueeze(0).to(device)
    lig_emb = lig_emb.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(inputs, lig_emb)
    
    z_slice = inputs.shape[2] // 2
    
    img_input = inputs[0, 0, :, :, z_slice].cpu().numpy()
    img_mask  = inputs[0, 1, :, :, z_slice].cpu().numpy()
    img_pred  = pred[0, 0, :, :, z_slice].cpu().numpy()
    img_true  = target[0, :, :, z_slice].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_input, cmap="gray"); axes[0].set_title("Input Density")
    axes[1].imshow(img_mask, cmap="Blues", alpha=0.7); axes[1].set_title("Protein Mask")
    axes[2].imshow(img_pred, cmap="magma"); axes[2].set_title(f"Prediction (Epoch {epoch})")
    axes[3].imshow(img_true, cmap="magma"); axes[3].set_title("Ground Truth")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"visual_epoch_{epoch:03d}.png")
    plt.close()

def plot_metrics(history, save_path):
    """Plots Loss, LR, Gradient Norm, and Masked MSE."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Global Loss
    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axs[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Global Loss (MSE + SSIM)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Masked MSE
    axs[0, 1].plot(epochs, history['masked_mse'], color='red')
    axs[0, 1].set_title('Masked MSE (Error INSIDE Pocket)')
    axs[0, 1].set_yscale('log')
    axs[0, 1].grid(True)

    # 3. Gradient Norm
    axs[1, 0].plot(epochs, history['grad_norm'], color='green')
    axs[1, 0].set_title('Gradient Norm (Stability)')
    axs[1, 0].set_yscale('log') 
    axs[1, 0].grid(True)

    # 4. Learning Rate
    axs[1, 1].plot(epochs, history['lr'], color='orange')
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ----------------------------
# HELPER: 3D EXPORT (NEW)
# ----------------------------
def save_mrc_samples(model, subset, set_name, device, num_samples=5):
    """
    Randomly selects samples from a Train/Val Subset, runs inference,
    and saves the Input, Prediction, and Ground Truth as .mrc files.
    """
    model.eval()
    output_dir = RESULTS_DIR / set_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # subset is a torch.utils.data.Subset
    # subset.indices gives the indices in the ORIGINAL dataset
    indices_to_sample = random.sample(subset.indices, min(num_samples, len(subset)))
    
    # Access the original dataset logic to get metadata (PDB ID, SMILES)
    # We open the H5 file freshly here to avoid pickling issues
    with h5py.File(HDF5_FILE, 'r') as f:
        print(f"--- Exporting {len(indices_to_sample)} {set_name} samples to {output_dir} ---")
        
        for idx in indices_to_sample:
            # 1. Load Data
            exp_density = f['exp_density'][idx] # (96,96,96)
            protein_mask = f['maps'][idx][0]
            ground_truth = f['ground_truth_maps'][idx]
            pdb_id = f['pdb_ids'][idx].decode('utf-8')
            smiles = f['ligand_smiles'][idx].decode('utf-8')

            # 2. Prepare Inputs
            input_numpy = np.stack([exp_density, protein_mask], axis=0)
            input_tensor = torch.from_numpy(input_numpy).unsqueeze(0).float().to(device)
            
            # Generate Fingerprint (reusing logic essentially)
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp_arr = np.array(fp, dtype=np.float32)
                lig_emb = torch.from_numpy(fp_arr).unsqueeze(0).float().to(device)
            except:
                lig_emb = torch.zeros(1, 1024).float().to(device)

            # 3. Inference
            with torch.no_grad():
                pred_tensor = model(input_tensor, lig_emb)
            
            # 4. Save .mrc Files
            base_name = f"{set_name}_{pdb_id}"
            
            # Save Input
            with mrcfile.new(output_dir / f"{base_name}_input.mrc", overwrite=True) as mrc:
                mrc.set_data(exp_density.astype(np.float32))
                mrc.voxel_size = 1.0
            
            # Save Prediction
            with mrcfile.new(output_dir / f"{base_name}_pred.mrc", overwrite=True) as mrc:
                mrc.set_data(pred_tensor.cpu().numpy().squeeze().astype(np.float32))
                mrc.voxel_size = 1.0
                
            # Save Ground Truth
            with mrcfile.new(output_dir / f"{base_name}_gt.mrc", overwrite=True) as mrc:
                mrc.set_data(ground_truth.astype(np.float32))
                mrc.voxel_size = 1.0
            
            print(f"   Saved {base_name}")


# ----------------------------
# DATASET CLASS
# ----------------------------
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

        exp_density = self.h5_file['exp_density'][idx]
        protein_mask = self.h5_file['maps'][idx][0]
        
        input_tensor = np.stack([exp_density, protein_mask], axis=0)
        
        target = self.h5_file['ground_truth_maps'][idx]
        target = np.expand_dims(target, axis=0) 

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
    
    if not HDF5_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {HDF5_FILE}")

    dataset = CryoEMDataset(HDF5_FILE)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SCUNet(in_nc=2, ligand_dim=1024).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    try:
        criterion = CustomLoss()
    except NameError:
        print("CustomLoss not found, using MSELoss.")
        criterion = nn.MSELoss()

    history = {
        'train_loss': [], 'val_loss': [], 
        'lr': [], 'grad_norm': [], 'masked_mse': []
    }

    best_loss = float('inf')

    print("--- Starting Training ---")
    try:
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
                
                grad_norm = get_grad_norm(model)
                total_grad_norm += grad_norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                t_loss += loss.item()

            # === VALIDATION ===
            model.eval()
            v_loss = 0
            total_masked_mse = 0
            
            with torch.no_grad():
                for inputs, lig_emb, targets in val_loader:
                    inputs, lig_emb, targets = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE)
                    preds = model(inputs, lig_emb)
                    
                    v_loss += criterion(preds, targets).item()
                    
                    # MASKED MSE
                    mask = inputs[:, 1:2, :, :, :] 
                    active_region = mask > 0.1
                    squared_diff = (preds - targets) ** 2
                    
                    if active_region.sum() > 0:
                        masked_mse = squared_diff[active_region].mean().item()
                    else:
                        masked_mse = squared_diff.mean().item()
                    
                    total_masked_mse += masked_mse

            # === AVERAGES & LOGGING ===
            avg_t = t_loss / len(train_loader)
            avg_v = v_loss / len(val_loader)
            avg_grad = total_grad_norm / len(train_loader)
            avg_masked = total_masked_mse / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            scheduler.step(avg_v)
            
            history['train_loss'].append(avg_t)
            history['val_loss'].append(avg_v)
            history['lr'].append(current_lr)
            history['grad_norm'].append(avg_grad)
            history['masked_mse'].append(avg_masked)

            print(f"Epoch {epoch+1:03d} | Val Loss: {avg_v:.5f} | Masked MSE: {avg_masked:.5f} | Grad: {avg_grad:.2f}")
            
            plot_metrics(history, save_path=PLOT_DIR / "training_metrics.png")
            
            # Save visual check every epoch
            save_visual_check(model, dataset, epoch + 1, DEVICE)

            if avg_v < best_loss:
                best_loss = avg_v
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
                print(f"  --> Best Model Saved")

            torch.save(model.state_dict(), CHECKPOINT_DIR / "latest_checkpoint.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Proceeding to 3D Export...")

    # --- 3D EXPORT STEP (Runs after training finishes) ---
    print("\n==================================================")
    print("TRAINING FINISHED. EXPORTING 3D SAMPLES FOR CHECKING")
    print("==================================================")
    
    # Load the best model to ensure we export the best results
    print("Loading Best Model for Export...")
    best_path = CHECKPOINT_DIR / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    
    # Save 5 samples from TRAIN set
    save_mrc_samples(model, train_data, "train", DEVICE, num_samples=5)
    
    # Save 5 samples from VAL set
    save_mrc_samples(model, val_data, "val", DEVICE, num_samples=5)
    
    print("\nDone! Check 'results/mrc_visuals/' for 3D files.")

if __name__ == "__main__":
    main()