import os
# --- CRITICAL FIX FOR HDF5 LOCKING ---
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import glob
import numpy as np
import threading
import h5py
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
# Removed Bio.PDB imports since we are using Gemmi now
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

# ----------------------------
# PARAMETERS
# ----------------------------
TEST_LIMIT = 10           # How many tasks to run
MAX_WORKERS = 8           # Number of threads

# --- CONFIGURABLE FILTER SWITCH ---
ONLY_TEST_AMINO_ACIDS = True  

AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

TARGET_VOXEL_SIZE = 0.5   
GRID_SIZE = 96            
SYNTHETIC_RESOLUTION = 4.0 
NUM_CHANNELS = 2

h5_lock = threading.Lock()
global_write_index = 0

# ----------------------------
# HELPER FUNCTIONS (Adapted for Gemmi)
# ----------------------------

def generate_synthetic_map(atoms, origin_angstroms, box_size, voxel_size, resolution):
    """
    Adapted to accept GEMMI atoms directly.
    """
    grid = np.zeros((box_size, box_size, box_size), dtype=np.float32)
    count = 0
    
    for atom in atoms:
        # Gemmi element handling
        elem = atom.element.name.upper()
        if elem == 'H': continue 
        
        # Gemmi position handling
        pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        
        rel_pos = (pos - origin_angstroms) / voxel_size
        idx = np.round(rel_pos).astype(np.int32)
        
        if (0 <= idx[0] < box_size and 0 <= idx[1] < box_size and 0 <= idx[2] < box_size):
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

# ----------------------------
# PROCESSOR (NOW USING GEMMI DIRECTLY)
# ----------------------------
def process_task(task, h5file, pbar):
    global global_write_index
    pdb_id, struct_path, map_path, target_ccd, target_smiles = task
    
    output_subdir = PROCESSED_DIR / pdb_id.lower()
    output_subdir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. READ STRUCTURE WITH GEMMI (No Biopython)
        # Gemmi reads CIF/PDB and preserves the HETATM vs ATOM flags perfectly.
        st = gemmi.read_structure(str(struct_path))
        
        # We iterate over the first model
        model = st[0]
        
        # 2. SEPARATE LIGANDS (HETATM) AND PROTEIN (ATOM)
        target_residues = []
        protein_atoms = []

        for chain in model:
            for res in chain:
                # 
                # Gemmi flag: 'H' = HETATM, 'A' = ATOM
                
                # Check for Target Ligand (Must be HETATM)
                if res.name == target_ccd and res.het_flag == 'H':
                    target_residues.append(res)
                
                # Check for Protein (Must be ATOM)
                elif res.het_flag == 'A':
                    for atom in res:
                        protein_atoms.append(atom)

        if not target_residues: return

        # 3. READ MAP
        m = gemmi.read_ccp4_map(str(map_path))
        m.setup(0.0, gemmi.MapSetup.Full)
        grid = resample_em_map(m.grid, TARGET_VOXEL_SIZE)
        
        for i, lig_res in enumerate(target_residues, 1):
            try:
                # Convert Gemmi residue atoms to a list
                ligand_atoms = list(lig_res)
                all_atoms = protein_atoms + ligand_atoms
                
                # Calculate Centroid (using Gemmi positions)
                lig_coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in ligand_atoms])
                centroid = np.mean(lig_coords, axis=0)
                
                center_idx = coord_to_grid_index(centroid, grid)
                half = GRID_SIZE // 2
                start_idx = center_idx - half
                start_idx = np.clip(start_idx, [0, 0, 0], np.array(grid.shape) - GRID_SIZE)

                density = grid.get_subarray(start_idx.tolist(), [GRID_SIZE, GRID_SIZE, GRID_SIZE]).astype(np.float32)
                
                frac_origin = gemmi.Fractional(start_idx[0]/grid.nu, start_idx[1]/grid.nv, start_idx[2]/grid.nw)
                phys_origin_vec = grid.unit_cell.orthogonalize(frac_origin)
                phys_origin = np.array([phys_origin_vec.x, phys_origin_vec.y, phys_origin_vec.z], dtype=np.float32)

                # Generate GT using updated Gemmi-compatible helper
                synthetic_density = generate_synthetic_map(all_atoms, phys_origin, GRID_SIZE, TARGET_VOXEL_SIZE, SYNTHETIC_RESOLUTION)
                
                synthetic_density = (synthetic_density - np.mean(synthetic_density)) / (np.std(synthetic_density) + 1e-6)
                density = (density - np.mean(density)) / (np.std(density) + 1e-6)

                data = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE), np.float32)
                
                # Channel 0: Protein (Gemmi Atoms)
                for atom in protein_atoms:
                    pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    v = coord_to_grid_index(pos, grid) - start_idx
                    if np.all((v >= 0) & (v < GRID_SIZE)): data[0, v[0], v[1], v[2]] = 1.0
                
                # Channel 1: Ligand (Gemmi Atoms)
                for atom in ligand_atoms:
                    pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    v = coord_to_grid_index(pos, grid) - start_idx
                    if np.all((v >= 0) & (v < GRID_SIZE)): data[1, v[0], v[1], v[2]] = 1.0
                
                mask = (data[1] > 0).astype(np.uint8)

                out_mrc = output_subdir / f"ml_map_{target_ccd}_{i}.mrc"
                save_mrc_with_origin(density.T, out_mrc, TARGET_VOXEL_SIZE, phys_origin)
                
                out_syn_mrc = output_subdir / f"ml_gt_{target_ccd}_{i}.mrc"
                save_mrc_with_origin(synthetic_density.T, out_syn_mrc, TARGET_VOXEL_SIZE, phys_origin)
                
                # Save Clean PDB using Gemmi (Replaces PDBIO)
                pdb_out_path = output_subdir / f"{pdb_id}_{target_ccd}_{i}_clean.pdb"
                new_st = gemmi.Structure()
                new_model = gemmi.Model("1")
                new_chain = gemmi.Chain("A")
                new_chain.add_residue(lig_res) # Copy just the ligand residue
                new_model.add_chain(new_chain)
                new_st.add_model(new_model)
                new_st.write_pdb(str(pdb_out_path))

                with h5_lock:
                    idx = global_write_index
                    global_write_index += 1
                    
                    h5file['maps'][idx] = data
                    h5file['exp_density'][idx] = density
                    h5file['ground_truth_maps'][idx] = synthetic_density
                    h5file['masks'][idx] = mask
                    h5file['pdb_ids'][idx] = pdb_id.encode()
                    h5file['ligand_names'][idx] = target_ccd.encode()
                    h5file['ligand_smiles'][idx] = target_smiles.encode()
                    h5file['centroids'][idx] = centroid.astype(np.float32)
                    h5file['crop_start_voxel'][idx] = start_idx
                    h5file['physical_origin'][idx] = phys_origin

                pbar.set_description(f"Processing (Saved: {global_write_index})")

            except Exception as e:
                pass 

    except Exception as e:
        print(f"Error {pdb_id}: {e}")
    finally:
        pbar.update(1)

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_H5 = PROCESSED_DIR / "ml_dataset_TEST.h5"

    if not METADATA_FILE.exists():
        print("Error: Metadata file missing.")
        exit()
        
    loaded = np.load(METADATA_FILE, allow_pickle=True)
    metadata = loaded["data"].item() if 'data' in loaded else loaded
    
    meta_lookup = {
        pid.lower(): (ccds, smis) 
        for pid, ccds, smis in zip(metadata["pdb_ids"], metadata["ligand_names"], metadata["ligand_smiles"])
    }

    # --- SCANNING WITH CONFIGURABLE FILTER ---
    if ONLY_TEST_AMINO_ACIDS:
        print(f"Scanning for {TEST_LIMIT} samples containing **AMINO ACID** ligands...")
    else:
        print(f"Scanning for {TEST_LIMIT} random samples (Filter OFF)...")
    
    existing_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    random.shuffle(existing_dirs)

    valid_tasks = []
    
    for d in existing_dirs:
        if TEST_LIMIT and len(valid_tasks) >= TEST_LIMIT:
            break
            
        pdb_id = d.name.lower()
        if pdb_id not in meta_lookup: continue
            
        struct_files = list(d.glob("*.cif"))
        if not struct_files: struct_files = list(d.glob("*.pdb"))
        map_files = list(d.glob("*.map.gz"))
        
        if struct_files and map_files:
            possible_ccds, possible_smiles = meta_lookup[pdb_id]
            
            for ccd, smi in zip(possible_ccds, possible_smiles):
                
                # --- FILTER LOGIC ---
                if ONLY_TEST_AMINO_ACIDS:
                    if ccd not in AMINO_ACIDS:
                        continue 
                
                valid_tasks.append((pdb_id, struct_files[0], map_files[0], ccd, smi))
                
                if len(valid_tasks) >= TEST_LIMIT: 
                    break 

    print(f"Found {len(valid_tasks)} tasks.")
    
    # --- PROCESSING ---
    ESTIMATED_MAX = len(valid_tasks) * 10 
    
    with h5py.File(OUTPUT_H5, "w") as h5:
        h5.create_dataset("maps", (ESTIMATED_MAX, NUM_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="float32", chunks=True)
        h5.create_dataset("exp_density", (ESTIMATED_MAX, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="float32", chunks=True)
        h5.create_dataset("ground_truth_maps", (ESTIMATED_MAX, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="float32", chunks=True)
        h5.create_dataset("masks", (ESTIMATED_MAX, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype="uint8", chunks=True)
        h5.create_dataset("pdb_ids", (ESTIMATED_MAX,), dtype=h5py.string_dtype(), chunks=True)
        h5.create_dataset("ligand_names", (ESTIMATED_MAX,), dtype=h5py.string_dtype(), chunks=True)
        h5.create_dataset("ligand_smiles", (ESTIMATED_MAX,), dtype=h5py.string_dtype(), chunks=True)
        h5.create_dataset("centroids", (ESTIMATED_MAX, 3), dtype="float32", chunks=True)
        h5.create_dataset("crop_start_voxel", (ESTIMATED_MAX, 3), dtype="int32", chunks=True)
        h5.create_dataset("physical_origin", (ESTIMATED_MAX, 3), dtype="float32", chunks=True)

        print("Starting Processing...")

        with tqdm(total=len(valid_tasks), desc="Processing") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_task, task, h5, pbar) for task in valid_tasks]
                for future in futures:
                    future.result()

        final_count = global_write_index
        print(f"\nComplete. Saved {final_count} instances.")
        for key in h5.keys():
            h5[key].resize(final_count, axis=0)

    print(f"Dataset saved: {OUTPUT_H5}")