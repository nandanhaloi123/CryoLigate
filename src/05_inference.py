import argparse
import torch
import mrcfile
import gemmi
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from pathlib import Path
import sys

# Add path to find architecture.py
sys.path.append(str(Path(__file__).resolve().parent))
from architecture import SCUNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Missing Helper Function ---
def get_fingerprint(smiles, dim=1024):
    """Generates the ligand fingerprint (must match training logic)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(dim, dtype=torch.float32)
        # Radius 2, 1024 bits (Matches your training config)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dim)
        arr = np.array(fp, dtype=np.float32)
        return torch.from_numpy(arr)
    except Exception as e:
        print(f"Warning: SMILES error: {e}")
        return torch.zeros(dim, dtype=torch.float32)

def preprocess(map_path, target_voxel=1.0, box_size=96):
    """
    Reads, resamples, normalizes, and crops the map.
    NOTE: target_voxel must match what you trained on (default 1.0).
    """
    m = gemmi.read_ccp4_map(map_path)
    m.setup(0.0, gemmi.MapSetup.Full)
    
    # 1. Resample to Target Voxel Size (Training was 1.0A)
    # We create a new grid with the correct dimensions to achieve target_voxel
    new_nu = int(m.grid.unit_cell.a / target_voxel)
    new_nv = int(m.grid.unit_cell.b / target_voxel)
    new_nw = int(m.grid.unit_cell.c / target_voxel)
    
    new_grid = gemmi.FloatGrid(new_nu, new_nv, new_nw)
    new_grid.set_unit_cell(gemmi.UnitCell(
        new_nu * target_voxel, 
        new_nv * target_voxel, 
        new_nw * target_voxel, 
        90, 90, 90
    ))
    
    gemmi.interpolate_grid(new_grid, m.grid, gemmi.Transform(), order=2)
    
    arr = np.array(new_grid, copy=False)
    
    # 2. Normalize (Standardization)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    
    # 3. Center Crop (CAUTION: Assumes ligand is in the center of the map)
    cx, cy, cz = np.array(arr.shape) // 2
    half = box_size // 2
    
    # Pad if map is smaller than box
    pad_x = max(0, half - cx) + max(0, half - (arr.shape[0]-cx))
    pad_y = max(0, half - cy) + max(0, half - (arr.shape[1]-cy))
    pad_z = max(0, half - cz) + max(0, half - (arr.shape[2]-cz))
    
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        arr = np.pad(arr, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
        cx, cy, cz = np.array(arr.shape) // 2

    crop = arr[cx-half:cx+half, cy-half:cy+half, cz-half:cz+half]
    
    return torch.from_numpy(crop).float(), target_voxel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="Path to input .mrc/.map file")
    parser.add_argument("--smiles", required=True, help="SMILES string of the ligand")
    parser.add_argument("--out", default="pred.mrc", help="Output filename")
    parser.add_argument("--model_path", default="../checkpoints/best_model.pth")
    args = parser.parse_args()

    print(f"--- Inference on {DEVICE} ---")

    # 1. Load Model
    model = SCUNet(in_nc=2, ligand_dim=1024).to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model not found at {args.model_path}")
        return
    model.eval()

    # 2. Preprocess Input
    # Returns (96, 96, 96) tensor
    vol_tensor, voxel_size = preprocess(args.map, target_voxel=1.0) 
    
    # 3. Handle Channels
    # Channel 0: Experimental Density (The input map)
    # Channel 1: Protein Mask. 
    # Since we don't have a mask, we create a "Pseudo-Mask" by thresholding.
    # Logic: High density (> 1 sigma) is likely protein/occupancy.
    mask_tensor = (vol_tensor > 1.0).float() 
    
    # Stack -> (1, 2, 96, 96, 96)
    inputs = torch.stack([vol_tensor, mask_tensor], dim=0).unsqueeze(0).to(DEVICE)

    # 4. Prepare Ligand
    emb = get_fingerprint(args.smiles, 1024).unsqueeze(0).to(DEVICE)
    
    print(f"Map shape: {inputs.shape} | Ligand: {args.smiles[:10]}...")

    # 5. Run Inference
    with torch.no_grad():
        out = model(inputs, emb)
    
    # 6. Save Output
    output_data = out.squeeze().cpu().numpy()
    
    with mrcfile.new(args.out, overwrite=True) as mrc:
        mrc.set_data(output_data.astype(np.float32))
        mrc.voxel_size = voxel_size
    
    print(f"Saved prediction to: {args.out}")

if __name__ == "__main__":
    main()