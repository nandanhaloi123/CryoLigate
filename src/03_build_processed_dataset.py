import os
# --- CRITICAL FIX FOR HDF5 LOCKING ---
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import glob
import numpy as np
import threading
import h5py
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
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "pdb_em_metadata.npz"

# ----------------------------
# PARAMETERS
# ----------------------------
TARGET_VOXEL_SIZE = 0.5   
GRID_SIZE = 96            
SYNTHETIC_RESOLUTION = 4.0 
NUM_CHANNELS = 2
MAX_WORKERS = 8
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
    """
    Generates a map using scipy.ndimage.gaussian_filter.
    """
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

    # Apply Gaussian Filter
    density = scipy.ndimage.gaussian_filter(grid, sigma=sigma_pixels, mode='constant', cval=0.0)

    # Apply Soft Saturation
    peak_approx = np.max(density)
    if peak_approx > 0:
        density = np.tanh(density / (peak_approx * 0.3 + 1e-6))

    return density

# ----------------------------
# PROCESS FUNCTION
# ----------------------------
def process_ligand_multichannel(task, h5file, index):
    pdb_dir, pdb_file, pdb_id, ligand_ccd, ligand_smiles, ligand_idx = task
    
    output_subdir = PROCESSED_DIR / pdb_id.lower()
    output_subdir.mkdir(parents=True, exist_ok=True)

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("struct", pdb_file)
        residues = list(structure.get_residues())

        protein_atoms = get_atoms([r for r in residues if r.id[0] == " "])
        ligand_residues = [r for r in residues if r.resname == ligand_ccd]
        
        if ligand_idx > len(ligand_residues): 
            return
            
        target_ligand = ligand_residues[ligand_idx - 1]
        ligand_atoms = list(target_ligand.get_atoms())
        all_atoms = protein_atoms + ligand_atoms

        # Save Clean PDB (for reference)
        pdb_out_path = output_subdir / f"{pdb_id}_{ligand_ccd}_{ligand_idx}_clean.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_out_path), select=SelectComplex(target_ligand))

        # Experimental Map Processing
        em_maps = glob.glob(os.path.join(pdb_dir, "EMD-*.map.gz"))
        if not em_maps: return
        
        m = gemmi.read_ccp4_map(em_maps[0])
        m.setup(0.0, gemmi.MapSetup.Full)
        grid = resample_em_map(m.grid, TARGET_VOXEL_SIZE)

        centroid = np.mean([a.coord for a in ligand_atoms], axis=0)
        center_idx = coord_to_grid_index(centroid, grid)
        half = GRID_SIZE // 2
        
        start_idx = center_idx - half
        start_idx = np.clip(start_idx, [0, 0, 0], np.array(grid.shape) - GRID_SIZE)

        # Extract Density
        density = grid.get_subarray(start_idx.tolist(), [GRID_SIZE, GRID_SIZE, GRID_SIZE]).astype(np.float32)
        
        # --- CALCULATE PHYSICAL ORIGIN (Crucial for Alignment) ---
        frac_origin = gemmi.Fractional(start_idx[0]/grid.nu, start_idx[1]/grid.nv, start_idx[2]/grid.nw)
        phys_origin_vec = grid.unit_cell.orthogonalize(frac_origin)
        phys_origin = np.array([phys_origin_vec.x, phys_origin_vec.y, phys_origin_vec.z], dtype=np.float32)

        # Generate Ground Truth
        synthetic_density = generate_synthetic_map(
            all_atoms, 
            phys_origin, 
            GRID_SIZE, 
            TARGET_VOXEL_SIZE,
            SYNTHETIC_RESOLUTION
        )
        
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
            # Save Origin for Training alignment
            h5file['physical_origin'][index] = phys_origin

        # --- CORRECTED: SAVE MRCs USING UTILS (WITH ORIGIN) ---
        # We pass density.T to fix axis order for visualization tools like ChimeraX
        out_mrc = output_subdir / f"ml_map_{ligand_ccd}_{ligand_idx}.mrc"
        save_mrc_with_origin(density.T, out_mrc, TARGET_VOXEL_SIZE, phys_origin)

        out_syn_mrc = output_subdir / f"ml_gt_{ligand_ccd}_{ligand_idx}.mrc"
        save_mrc_with_origin(synthetic_density.T, out_syn_mrc, TARGET_VOXEL_SIZE, phys_origin)

    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        # import traceback
        # traceback.print_exc()

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_H5 = PROCESSED_DIR / "ml_dataset.h5"

    loaded = np.load(METADATA_FILE, allow_pickle=True)
    metadata = loaded["data"].item()

    tasks = []
    print("Scanning PDBs for ligand instances...")
    
    for pdb_id, ccds, smiles in zip(metadata["pdb_ids"], metadata["ligand_names"], metadata["ligand_smiles"]):
        pdb_dir = RAW_DATA_DIR / pdb_id.lower()
        if not pdb_dir.exists(): continue
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        if not pdb_files: continue
        
        if not glob.glob(os.path.join(pdb_dir, "EMD-*.map.gz")): continue
        
        try:
            parser = PDBParser(QUIET=True)
            s = parser.get_structure("temp", pdb_files[0])
            target_ccd = ccds[0]
            
            lig_instances = [r for r in s.get_residues() if r.resname == target_ccd]
            count = len(lig_instances)
            
            if count == 0:
                print(f"  [Skip] {pdb_id}: Ligand {target_ccd} not found in PDB.")
                continue
            
            print(f"  [Add]  {pdb_id}: Found {count} instances of {target_ccd}")

            for i in range(1, count + 1):
                tasks.append((str(pdb_dir), pdb_files[0], pdb_id, target_ccd, smiles[0], i))

        except Exception as e:
            print(f"  [Error] Scanning {pdb_id}: {e}")

    print(f"\nFound {len(tasks)} total training samples.")
    
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

        print(f"Starting Processing (Output -> {PROCESSED_DIR})...")
        thread_map(
            lambda x: process_ligand_multichannel(x[1], h5, x[0]),
            list(enumerate(tasks)),
            max_workers=MAX_WORKERS,
            desc="Building Dataset"
        )

    print("Dataset generated successfully.")