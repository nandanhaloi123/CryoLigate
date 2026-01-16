import os
# --- CRITICAL FIX FOR HDF5 LOCKING ---
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import glob
import numpy as np
import threading
import h5py
import mrcfile
import gemmi
import scipy.ndimage 
from pathlib import Path
from tqdm.contrib.concurrent import thread_map
from Bio.PDB import PDBParser, PDBIO, Select

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

# Resolution: 4.0A ensures atoms overlap smoothly without gaps.
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

def resample_em_map(grid, target_voxel):
    new_size = [
        int(round(grid.unit_cell.a / target_voxel)),
        int(round(grid.unit_cell.b / target_voxel)),
        int(round(grid.unit_cell.c / target_voxel))
    ]
    new_grid = gemmi.FloatGrid(*new_size)
    new_grid.set_unit_cell(gemmi.UnitCell(
        new_size[0] * target_voxel,
        new_size[1] * target_voxel,
        new_size[2] * target_voxel,
        90, 90, 90
    ))
    gemmi.interpolate_grid(new_grid, grid, gemmi.Transform(), order=2)
    return new_grid

def coord_to_grid_index(coord, grid):
    pos = gemmi.Position(coord[0], coord[1], coord[2])
    fractional = grid.unit_cell.fractionalize(pos)
    return np.array([
        int(round(fractional.x * grid.nu)),
        int(round(fractional.y * grid.nv)),
        int(round(fractional.z * grid.nw)),
    ], dtype=np.int32)

# ----------------------------
# ROBUST GAUSSIAN GENERATOR (SCIPY VERSION)
# ----------------------------
def generate_synthetic_map(atoms, origin_angstroms, box_size, voxel_size, resolution):
    """
    Generates a map using scipy.ndimage.gaussian_filter.
    1. Places point sources (delta functions) at atom centers.
    2. Blurs the entire grid with a Gaussian.
    3. Applies tanh saturation to fix the 'hot core' issue.
    """
    # 1. Create Empty Grid
    grid = np.zeros((box_size, box_size, box_size), dtype=np.float32)
    
    # 2. Map Atoms to Grid Indices
    # We populate the grid with 1.0s where atoms are.
    # We check bounds to avoid errors.
    count = 0
    for atom in atoms:
        elem = atom.element.upper().strip()
        if elem == 'H': continue # Skip hydrogens

        # Convert World Coord -> Box Index
        rel_pos = (atom.coord - origin_angstroms) / voxel_size
        idx = np.round(rel_pos).astype(np.int32)
        
        if (0 <= idx[0] < box_size and 
            0 <= idx[1] < box_size and 
            0 <= idx[2] < box_size):
            grid[idx[0], idx[1], idx[2]] += 1.0
            count += 1

    if count == 0:
        return grid

    # 3. Calculate Sigma for Gaussian Filter
    # ChimeraX Formula: sigma = 0.225 * resolution
    sigma_angstrom = 0.225 * resolution
    sigma_pixels = sigma_angstrom / voxel_size

    # 4. Apply Gaussian Filter (The "Blur")
    # This creates the smooth blobs and naturally sums overlapping densities.
    density = scipy.ndimage.gaussian_filter(grid, sigma=sigma_pixels, mode='constant', cval=0.0)

    # 5. Apply Soft Saturation (The Fix for "Unevenness")
    # Tanh compresses high values (core) while keeping low values (surface) linear.
    # We scale it so that a single atom (peak ~0.02 depending on sigma) is well visible.
    
    # Heuristic: Normalize so the max is around 1.0, then saturate outliers.
    peak_approx = np.max(density)
    if peak_approx > 0:
        # Scale so the typical 'surface' density is around 0.5 - 1.0
        # This pushes the 'core' density into the flat part of the tanh curve.
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
        ligand_residues = [r for r in residues if r.resname == ligand_ccd and r.id[0] != " "]
        
        if ligand_idx > len(ligand_residues): return
        target_ligand = ligand_residues[ligand_idx - 1]
        ligand_atoms = list(target_ligand.get_atoms())
        all_atoms = protein_atoms + ligand_atoms

        # Save Clean PDB
        pdb_out_path = output_subdir / f"{pdb_id}_clean.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_out_path), select=SelectComplex(target_ligand))

        # Experimental Map
        em_maps = glob.glob(os.path.join(pdb_dir, "EMD-*.map.gz"))
        if not em_maps: return
        
        m = gemmi.read_ccp4_map(em_maps[0])
        m.setup(0.0, gemmi.MapSetup.Full)
        grid = resample_em_map(m.grid, TARGET_VOXEL_SIZE)

        centroid = np.mean([a.coord for a in ligand_atoms], axis=0)
        center_idx = coord_to_grid_index(centroid, grid)
        half = GRID_SIZE // 2
        
        start = center_idx - half
        start = np.clip(start, [0, 0, 0], np.array(grid.shape) - GRID_SIZE)

        # Get Experimental Density
        density = grid.get_subarray(start.tolist(), [GRID_SIZE, GRID_SIZE, GRID_SIZE]).astype(np.float32)
        
        frac_origin = gemmi.Fractional(start[0]/grid.nu, start[1]/grid.nv, start[2]/grid.nw)
        phys_origin_vec = grid.unit_cell.orthogonalize(frac_origin)
        phys_origin = np.array([phys_origin_vec.x, phys_origin_vec.y, phys_origin_vec.z])

        # --- GENERATE GROUND TRUTH (SCIPY VERSION) ---
        synthetic_density = generate_synthetic_map(
            all_atoms, 
            phys_origin, 
            GRID_SIZE, 
            TARGET_VOXEL_SIZE,
            SYNTHETIC_RESOLUTION
        )
        
        # Normalize (Standard Z-score for final output)
        syn_mean, syn_std = np.mean(synthetic_density), np.std(synthetic_density)
        synthetic_density = (synthetic_density - syn_mean) / (syn_std + 1e-6)

        exp_mean, exp_std = np.mean(density), np.std(density)
        density = (density - exp_mean) / (exp_std + 1e-6)

        # Masks
        data = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE), np.float32)
        for atom in protein_atoms:
            v = coord_to_grid_index(atom.coord, grid) - start
            if np.all((v >= 0) & (v < GRID_SIZE)):
                data[0, v[0], v[1], v[2]] = 1.0

        for atom in ligand_atoms:
            v = coord_to_grid_index(atom.coord, grid) - start
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
            h5file['crop_start_voxel'][index] = start

        # Save MRCs
        out_mrc = output_subdir / f"ml_map_{ligand_ccd}_{ligand_idx}.mrc"
        with mrcfile.new(str(out_mrc), overwrite=True) as mrc:
            mrc.set_data(np.ascontiguousarray(density.T)) 
            mrc.voxel_size = TARGET_VOXEL_SIZE
            mrc.header.origin.x = float(phys_origin[0])
            mrc.header.origin.y = float(phys_origin[1])
            mrc.header.origin.z = float(phys_origin[2])
            mrc.update_header_from_data()

        out_syn_mrc = output_subdir / f"ml_gt_{ligand_ccd}_{ligand_idx}.mrc"
        with mrcfile.new(str(out_syn_mrc), overwrite=True) as mrc:
            mrc.set_data(np.ascontiguousarray(synthetic_density.T))
            mrc.voxel_size = TARGET_VOXEL_SIZE
            mrc.header.origin.x = float(phys_origin[0])
            mrc.header.origin.y = float(phys_origin[1])
            mrc.header.origin.z = float(phys_origin[2])
            mrc.update_header_from_data()

    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        import traceback
        traceback.print_exc()

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_H5 = PROCESSED_DIR / "ml_dataset.h5"

    loaded = np.load(METADATA_FILE, allow_pickle=True)
    metadata = loaded["data"].item()

    tasks = []
    print("Preparing tasks...")
    for pdb_id, ccds, smiles in zip(metadata["pdb_ids"], metadata["ligand_names"], metadata["ligand_smiles"]):
        pdb_dir = RAW_DATA_DIR / pdb_id.lower()
        if not pdb_dir.exists(): continue
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        if not pdb_files: continue
        
        if not glob.glob(os.path.join(pdb_dir, "EMD-*.map.gz")): continue

        tasks.append((str(pdb_dir), pdb_files[0], pdb_id, ccds[0], smiles[0], 1))

    print(f"Found {len(tasks)} valid tasks.")
    
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

        print(f"Starting Processing (Output -> {PROCESSED_DIR})...")
        thread_map(
            lambda x: process_ligand_multichannel(x[1], h5, x[0]),
            list(enumerate(tasks)),
            max_workers=MAX_WORKERS,
            desc="Building Dataset"
        )

    print("Dataset generated successfully.")