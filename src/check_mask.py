import h5py
import numpy as np
from pathlib import Path
import sys

# Ensure we can find utils_common
sys.path.append(str(Path(__file__).resolve().parent))
from utils_common import save_mrc_with_origin

# Define paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
HDF5_FILE = DATA_DIR / "ml_dataset.h5"

if not HDF5_FILE.exists():
    raise FileNotFoundError(f"Could not find {HDF5_FILE}")

with h5py.File(HDF5_FILE, 'r') as f:
    idx = 0 
    
    # Load the Mask and Origin
    mask = f['masks'][idx]  # This is the Ligand Only Mask
    origin = f['physical_origin'][idx]
    pdb_id = f['pdb_ids'][idx].decode()
    
    print(f"Checking mask for PDB: {pdb_id}")
    
    # FIX: Wrap the string in Path()
    output_path = SCRIPT_DIR / "debug_ligand_mask.mrc"
    
    # Save it (Transposed .T for ChimeraX)
    save_mrc_with_origin(mask.T, output_path, 0.5, origin)
    
    print(f"Saved to: {output_path}")
    print("Open this in ChimeraX to verify it contains ONLY the ligand shape.")