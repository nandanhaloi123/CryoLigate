import os

# --- CRITICAL FIX FOR HDF5 LOCKING ---
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import glob
import numpy as np
import threading
import h5py
import pandas as pd
from pathlib import Path
from tqdm.contrib.concurrent import thread_map
from Bio.PDB import PDBParser, PDBIO, Select
import gemmi
import scipy.ndimage

# --- IMPORT SHARED UTILS ---
from utils_common import resample_em_map, coord_to_grid_index, save_mrc_with_origin

# ----------------------------
# PATH CONFIGURATION
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "pdb_em_metadata_balanced.npz"
EXCEL_REPORT = PROCESSED_DIR / "dataset_metadata.xlsx"

# ----------------------------
# PARAMETERS
# ----------------------------
TARGET_VOXEL_SIZE = 0.5   
GRID_SIZE = 96            
SYNTHETIC_RESOLUTION = 4.0 
NUM_CHANNELS = 2
MAX_WORKERS = 8

# FILTERS
MIN_QUALITY_SCORE = 0.3    # CC Score Cutoff
MAX_OLIGOMERIC_STATE = 10  # Skip if >= 10 chains

h5_lock = threading.Lock()

# ----------------------------
# HELPER CLASSES & FUNCTIONS
# ----------------------------
class SelectComplex(Select):
    def __init__(self, target_ligand):
        self.target = target_ligand
    def accept_residue(self, residue):
        return 1 if (residue == self.target or residue.id[0] == " ") else 0

def get_atoms(residues):
    atoms = []
    for r in residues:
        atoms.extend(list(r.get_atoms()))
    return atoms

def generate_synthetic_map(atoms, origin_angstroms, box_size, voxel_size, resolution):
    grid = np.zeros((box_size, box_size, box_size), dtype=np.float32)
    
    count = 0
    for atom in atoms:
        elem = atom.element.upper().strip()
        if elem == 'H': continue 

        rel_pos = (atom.coord - origin_angstroms) / voxel_size
        idx = np.round(rel_pos).astype(np.int32)
        
        if (0 <= idx[0] < box_size and 
            0 <= idx[1] < box_size and 
            0 <= idx[2] < box_size):
            grid[idx[0], idx[1], idx[2]] += 1.0
            count += 1

    if count == 0: return grid

    sigma_angstrom = 0.225 * resolution
    sigma_pixels = sigma_angstrom / voxel_size
    density = scipy.ndimage.gaussian_filter(grid, sigma=sigma_pixels, mode='constant', cval=0.0)

    peak_approx = np.max(density)
    if peak_approx > 0:
        density = np.tanh(density / (peak_approx * 0.3 + 1e-6))

    return density

def calculate_correlation(map_a, map_b):
    flat_a = map_a.flatten()
    flat_b = map_b.flatten()
    if np.std(flat_a) == 0 or np.std(flat_b) == 0:
        return 0.0
    return np.corrcoef(flat_a, flat_b)[0, 1]

# ----------------------------
# PROCESS FUNCTION
# ----------------------------
def process_ligand_multichannel(task, h5file, index):
    pdb_dir, pdb_file, map_file, pdb_id, ligand_ccd, ligand_smiles, ligand_idx = task
    
    output_subdir = PROCESSED_DIR / pdb_id.lower()
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Initialize Metadata Entry
    meta_entry = {
        "pdb_id": pdb_id,
        "ligand": ligand_ccd,
        "instance_id": ligand_idx,
        "cc_score": 0.0,
        "status": "Failed",
        "reason": ""
    }

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("struct", pdb_file)
        residues = list(structure.get_residues())

        protein_atoms = get_atoms([r for r in residues if r.id[0] == " "])
        ligand_residues = [r for r in residues if r.resname == ligand_ccd]
        
        if ligand_idx > len(ligand_residues): 
            meta_entry["reason"] = "Index out of bounds"
            return meta_entry
            
        target_ligand = ligand_residues[ligand_idx - 1]
        ligand_atoms = list(target_ligand.get_atoms())
        all_atoms = protein_atoms + ligand_atoms

        # 1. Map Processing
        m = gemmi.read_ccp4_map(map_file)
        m.setup(0.0, gemmi.MapSetup.Full)
        grid = resample_em_map(m.grid, TARGET_VOXEL_SIZE)

        # 2. Box Extraction
        centroid = np.mean([a.coord for a in ligand_atoms], axis=0)
        center_idx = coord_to_grid_index(centroid, grid)
        half = GRID_SIZE // 2
        start_idx = center_idx - half
        start_idx = np.clip(start_idx, [0, 0, 0], np.array(grid.shape) - GRID_SIZE)

        density = grid.get_subarray(start_idx.tolist(), [GRID_SIZE, GRID_SIZE, GRID_SIZE]).astype(np.float32)
        
        frac_origin = gemmi.Fractional(start_idx[0]/grid.nu, start_idx[1]/grid.nv, start_idx[2]/grid.nw)
        phys_origin_vec = grid.unit_cell.orthogonalize(frac_origin)
        phys_origin = np.array([phys_origin_vec.x, phys_origin_vec.y, phys_origin_vec.z], dtype=np.float32)

        synthetic_density = generate_synthetic_map(all_atoms, phys_origin, GRID_SIZE, TARGET_VOXEL_SIZE, SYNTHETIC_RESOLUTION)
        
        # 3. Quality Check (CC)
        ligand_only_density = generate_synthetic_map(ligand_atoms, phys_origin, GRID_SIZE, TARGET_VOXEL_SIZE, SYNTHETIC_RESOLUTION)
        quality_score = calculate_correlation(ligand_only_density, density)
        
        meta_entry["cc_score"] = round(quality_score, 4)

        if quality_score < MIN_QUALITY_SCORE:
            meta_entry["status"] = "Skipped"
            meta_entry["reason"] = f"Low CC ({quality_score:.2f} < {MIN_QUALITY_SCORE})"
            return meta_entry

        # Normalize
        synthetic_density = (synthetic_density - np.mean(synthetic_density)) / (np.std(synthetic_density) + 1e-6)
        density = (density - np.mean(density)) / (np.std(density) + 1e-6)

        # Masks
        data = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE), np.float32)
        for atom in protein_atoms:
            v = coord_to_grid_index(atom.coord, grid) - start_idx
            if np.all((v >= 0) & (v < GRID_SIZE)):
                data[0, v[0], v[1], v[2]] = 1.0

        for atom in ligand_atoms:
            v = coord_to_grid_index(atom.coord, grid) - start_idx
            if np.all((v >= 0) & (v < GRID_SIZE)):
                data[1, v[0], v[1], v[2]] = 1.0
        
        mask = (data[1] > 0).astype(np.uint8)

        # Save HDF5
        with h5_lock:
            h5file['maps'][index] = data
            h5file['exp_density'][index] = density
            h5file['ground_truth_maps'][index] = synthetic_density
            h5file['masks'][index] = mask
            h5file['pdb_ids'][index] = pdb_id.encode()
            h5file['ligand_names'][index] = ligand_ccd.encode()
            h5file['ligand_smiles'][index] = ligand_smiles.encode()
            h5file['centroids'][index] = centroid
            h5file['crop_start_voxel'][index] = start_idx
            h5file['physical_origin'][index] = phys_origin
            h5file['ligand_quality_scores'][index] = quality_score

        # Save Visuals
        out_mrc = output_subdir / f"ml_map_{ligand_ccd}_{ligand_idx}_Q{quality_score:.2f}.mrc"
        save_mrc_with_origin(density.T, out_mrc, TARGET_VOXEL_SIZE, phys_origin)
        
        out_syn_mrc = output_subdir / f"ml_gt_{ligand_ccd}_{ligand_idx}.mrc"
        save_mrc_with_origin(synthetic_density.T, out_syn_mrc, TARGET_VOXEL_SIZE, phys_origin)

        meta_entry["status"] = "Saved"
        return meta_entry

    except Exception as e:
        meta_entry["status"] = "Error"
        meta_entry["reason"] = str(e)
        return meta_entry

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_H5 = PROCESSED_DIR / "ml_dataset.h5"

    loaded = np.load(METADATA_FILE, allow_pickle=True)
    metadata = loaded["data"].item()
    
    # Extract metadata arrays
    all_pdb_ids = metadata["pdb_ids"]
    all_ccds = metadata["ligand_names"]
    all_smiles = metadata["ligand_smiles"]
    all_olig_states = metadata["olig_states"] # Required for filter

    tasks = []
    skipped_oligomer = 0

    print(f"Scanning PDBs (Filter: Oligomer < {MAX_OLIGOMERIC_STATE})...")
    
    for pdb_id, ccds, smiles, state_raw in zip(all_pdb_ids, all_ccds, all_smiles, all_olig_states):
        
        # --- 1. OLIGOMER FILTER ---
        try:
            state_val = int(state_raw)
            if state_val >= MAX_OLIGOMERIC_STATE:
                skipped_oligomer += 1
                continue
        except (ValueError, TypeError):
            # If state is undefined, we usually proceed, but check if you want to be strict
            pass
        # ---------------------------

        pdb_dir = RAW_DATA_DIR / pdb_id.lower()
        if not pdb_dir.exists(): continue
        
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        map_files = glob.glob(os.path.join(pdb_dir, "EMD-*.map.gz"))
        
        # --- 2. FILE EXISTENCE FILTER ---
        if not (pdb_files and map_files): continue
        
        try:
            current_pdb = pdb_files[0]
            current_map = map_files[0]

            parser = PDBParser(QUIET=True)
            s = parser.get_structure("temp", current_pdb)
            target_ccd = ccds[0]
            
            lig_instances = [r for r in s.get_residues() if r.resname == target_ccd]
            count = len(lig_instances)
            
            if count == 0: continue
            
            print(f"  [Add]  {pdb_id}: Found {count} instances of {target_ccd}")

            for i in range(1, count + 1):
                tasks.append((str(pdb_dir), current_pdb, current_map, pdb_id, target_ccd, smiles[0], i))

        except Exception as e:
            print(f"  [Error] Scanning {pdb_id}: {e}")

    print(f"\nSkipped {skipped_oligomer} entries due to high oligomeric state.")
    print(f"Found {len(tasks)} valid candidates ready for processing.")
    
    with h5py.File(OUTPUT_H5, "w") as h5:
        N = len(tasks)
        h5.create_dataset("maps", (N, NUM_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="float32")
        h5.create_dataset("exp_density", (N, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="float32")
        h5.create_dataset("ground_truth_maps", (N, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="float32")
        h5.create_dataset("masks", (N, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="uint8")
        h5.create_dataset("pdb_ids", (N,), dtype=h5py.string_dtype())
        h5.create_dataset("ligand_names", (N,), dtype=h5py.string_dtype())
        h5.create_dataset("ligand_smiles", (N,), dtype=h5py.string_dtype())
        h5.create_dataset("centroids", (N, 3), dtype="float32")
        h5.create_dataset("crop_start_voxel", (N, 3), dtype="int32")
        h5.create_dataset("physical_origin", (N, 3), dtype="float32")
        h5.create_dataset("ligand_quality_scores", (N,), dtype="float32")

        print(f"Starting Processing (CC > {MIN_QUALITY_SCORE})...")
        
        results = thread_map(
            lambda x: process_ligand_multichannel(x[1], h5, x[0]),
            list(enumerate(tasks)),
            max_workers=MAX_WORKERS,
            desc="Building Dataset"
        )

    # --- SAVE EXCEL REPORT ---
    print(f"Saving metadata report to {EXCEL_REPORT}...")
    
    clean_results = [r for r in results if r is not None]
    df = pd.DataFrame(clean_results)
    
    if not df.empty:
        cols = ["pdb_id", "ligand", "instance_id", "cc_score", "status", "reason"]
        df = df[cols]
        df.to_excel(EXCEL_REPORT, index=False)
        
        print("\nProcessing Summary:")
        print(df["status"].value_counts())
    else:
        print("No data processed!")

    print(f"\nFull report saved to: {EXCEL_REPORT}")