import argparse
import os
# --- CRITICAL FIX FOR HDF5 LOCKING ---
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import glob
import numpy as np
import threading
import h5py
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gemmi
import scipy.ndimage

# --- IMPORT SHARED UTILS ---
from utils_common import resample_em_map, coord_to_grid_index, save_mrc_with_origin

# ----------------------------
# ARGUMENT PARSING
# ----------------------------
parser = argparse.ArgumentParser(description="Distributed Dataset Building")
parser.add_argument("--node_id", type=int, default=0, help="Index of the current node")
parser.add_argument("--total_nodes", type=int, default=1, help="Total number of nodes")
args = parser.parse_args()

# ----------------------------
# PATH CONFIGURATION
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "pdb_em_metadata_balanced.npz"

# NEW: Output for stats
STATS_EXCEL = PROCESSED_DIR / f"processing_stats_node_{args.node_id}.xlsx"
DISTRIBUTION_PLOT = PROCESSED_DIR / f"final_distribution_node_{args.node_id}.png"

# ----------------------------
# PARAMETERS
# ----------------------------
TEST_LIMIT = None         
MAX_WORKERS = 4           
ONLY_TEST_AMINO_ACIDS = False  
AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}
TARGET_VOXEL_SIZE = 0.5 
GRID_SIZE = 64            # 32 Angstrom Box
SYNTHETIC_RESOLUTION = 4.0 
NUM_CHANNELS = 2

# Max diameter allowed. 
MAX_LIGAND_DIAMETER = (GRID_SIZE * TARGET_VOXEL_SIZE) - 4.0 

# --- EXPANDED IGNORE LIST ---
IGNORED_LIGANDS = {
    "HOH", "DOD", "WAT", 
    "NA", "MG", "CL", "ZN", "MN", "CA", "K", "FE", "CU", "NI", "CO", "CD", "HG", "IOD",
    "SO4", "PO4", "ACT", "GOL", "PEG", "EDO", "DMS", "FMT", "ACY",
    "OLA", "OLC", "MYR", "MYA", "PAL", "PLM", "STE",
    "POP", "POPC", "DLPE", "DPPC", "DMPC", 
    "BOG", "BGC", "DDM", "LMT", "NGP", "TRX", "LDA", "UD1", "HTG",
    "CHL", "CLR", "NAG", "MAN", "BMA", "FUC" 
}

h5_lock = threading.Lock()
stats_lock = threading.Lock()
global_write_index = 0

# --- STATS TRACKING CONTAINERS ---
# Format: { 'PDB_ID': {'saved_count': 0, 'removed_count': 0, 'saved_names': [], 'removed_names': []} }
processing_stats = {}
# Format: Counter for saved ligands to generate plot
saved_ligand_counter = {}

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def get_max_diameter(atoms):
    """Calculates the maximum distance between any two atoms in the list."""
    coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
    if len(coords) < 2: return 0.0
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    return np.max(dists)

def generate_synthetic_map(atoms, origin_angstroms, box_size, voxel_size, resolution):
    grid = np.zeros((box_size, box_size, box_size), dtype=np.float32)
    count = 0
    for atom in atoms:
        elem = atom.element.name.upper()
        if elem == 'H': continue 
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

def check_ligand_interference_strict(target_res, all_het_residues, phys_origin, box_dims_angstrom):
    box_min = phys_origin
    box_max = phys_origin + box_dims_angstrom
    buffer = 1.0 
    box_min -= buffer
    box_max += buffer

    for other_res in all_het_residues:
        if other_res is target_res: continue
        for atom in other_res:
            pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
            if np.all(pos >= box_min) and np.all(pos <= box_max):
                return True, f"{other_res.name} (Seq: {other_res.seqid.num}) found inside crop box"
    return False, None

def update_stats(pdb_id, status, ligand_name):
    """Thread-safe update of statistics."""
    with stats_lock:
        if pdb_id not in processing_stats:
            processing_stats[pdb_id] = {
                'saved_count': 0, 
                'removed_count': 0, 
                'saved_names': [], 
                'removed_names': []
            }
        
        if status == "SAVED":
            processing_stats[pdb_id]['saved_count'] += 1
            processing_stats[pdb_id]['saved_names'].append(ligand_name)
            
            # Update global counter for plotting
            if ligand_name in saved_ligand_counter:
                saved_ligand_counter[ligand_name] += 1
            else:
                saved_ligand_counter[ligand_name] = 1
                
        elif status == "REMOVED":
            processing_stats[pdb_id]['removed_count'] += 1
            processing_stats[pdb_id]['removed_names'].append(ligand_name)

# ----------------------------
# PROCESSOR
# ----------------------------
def process_task(task, h5file, pbar):
    global global_write_index
    pdb_id, struct_path, map_path, target_ccd, target_smiles = task
    
    if target_ccd in IGNORED_LIGANDS:
        update_stats(pdb_id, "REMOVED", f"{target_ccd} (Ignored List)")
        pbar.update(1)
        return

    output_subdir = PROCESSED_DIR / pdb_id.lower()
    output_subdir.mkdir(parents=True, exist_ok=True)

    try:
        st = gemmi.read_structure(str(struct_path))
        model = st[0]
        
        target_residues = []
        protein_atoms = []
        all_het_residues = [] 

        for chain in model:
            for res in chain:
                if res.het_flag == 'H':
                    if res.name not in ["HOH", "DOD", "WAT"]:
                        all_het_residues.append(res)
                    if res.name == target_ccd:
                        target_residues.append(res)
                elif res.het_flag == 'A':
                    for atom in res:
                        protein_atoms.append(atom)

        if not target_residues: 
            update_stats(pdb_id, "REMOVED", f"{target_ccd} (Not found in PDB)")
            pbar.update(1)
            return

        m = gemmi.read_ccp4_map(str(map_path))
        m.setup(0.0, gemmi.MapSetup.Full)
        grid = resample_em_map(m.grid, TARGET_VOXEL_SIZE)
        del m  
        
        for i, lig_res in enumerate(target_residues, 1):
            unique_lig_name = f"{target_ccd}_{i}"
            try:
                ligand_atoms = list(lig_res)
                
                # SIZE CHECK
                diameter = get_max_diameter(ligand_atoms)
                if diameter > MAX_LIGAND_DIAMETER:
                    print(f"[{pdb_id}] SKIPPED {unique_lig_name}: Too large")
                    update_stats(pdb_id, "REMOVED", f"{unique_lig_name} (Too Large)")
                    continue

                lig_coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in ligand_atoms])
                centroid = np.mean(lig_coords, axis=0)

                center_idx = coord_to_grid_index(centroid, grid)
                half = GRID_SIZE // 2
                start_idx = center_idx - half
                start_idx = np.clip(start_idx, [0, 0, 0], np.array(grid.shape) - GRID_SIZE)
                
                frac_origin = gemmi.Fractional(start_idx[0]/grid.nu, start_idx[1]/grid.nv, start_idx[2]/grid.nw)
                phys_origin_vec = grid.unit_cell.orthogonalize(frac_origin)
                phys_origin = np.array([phys_origin_vec.x, phys_origin_vec.y, phys_origin_vec.z], dtype=np.float32)
                
                box_dims = np.array([GRID_SIZE * TARGET_VOXEL_SIZE] * 3) 

                has_interference, reason = check_ligand_interference_strict(
                    lig_res, all_het_residues, phys_origin, box_dims
                )
                
                if has_interference:
                    print(f"[{pdb_id}] SKIPPED {unique_lig_name}: {reason}")
                    update_stats(pdb_id, "REMOVED", f"{unique_lig_name} (Interference)")
                    continue
                
                # --- GENERATE DATA ---
                all_atoms = protein_atoms + ligand_atoms
                density = grid.get_subarray(start_idx.tolist(), [GRID_SIZE, GRID_SIZE, GRID_SIZE]).astype(np.float32)
                synthetic_density = generate_synthetic_map(all_atoms, phys_origin, GRID_SIZE, TARGET_VOXEL_SIZE, SYNTHETIC_RESOLUTION)
                
                synthetic_density = (synthetic_density - np.mean(synthetic_density)) / (np.std(synthetic_density) + 1e-6)
                density = (density - np.mean(density)) / (np.std(density) + 1e-6)

                data = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE, GRID_SIZE), np.float32)
                
                for atom in protein_atoms:
                    pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    v = coord_to_grid_index(pos, grid) - start_idx
                    if np.all((v >= 0) & (v < GRID_SIZE)): data[0, v[0], v[1], v[2]] = 1.0
                
                for atom in ligand_atoms:
                    pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    v = coord_to_grid_index(pos, grid) - start_idx
                    if np.all((v >= 0) & (v < GRID_SIZE)): data[1, v[0], v[1], v[2]] = 1.0
                
                mask = (data[1] > 0).astype(np.uint8)

                # Save intermediate files
                out_mrc = output_subdir / f"ml_map_{unique_lig_name}.mrc"
                save_mrc_with_origin(density.T, out_mrc, TARGET_VOXEL_SIZE, phys_origin)
                out_syn_mrc = output_subdir / f"ml_gt_{unique_lig_name}.mrc"
                save_mrc_with_origin(synthetic_density.T, out_syn_mrc, TARGET_VOXEL_SIZE, phys_origin)
                
                pdb_out_path = output_subdir / f"{pdb_id}_{unique_lig_name}_clean.pdb"
                new_st = gemmi.Structure()
                new_model = gemmi.Model("1")
                new_chain = gemmi.Chain("A")
                new_chain.add_residue(lig_res)
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

                update_stats(pdb_id, "SAVED", target_ccd)
                pbar.set_description(f"Processing (Saved: {global_write_index})")

            except Exception:
                update_stats(pdb_id, "REMOVED", f"{unique_lig_name} (Error)")
                pass 

    except Exception as e:
        print(f"Error {pdb_id}: {e}")
        update_stats(pdb_id, "REMOVED", "Entire PDB Failed")
    finally:
        pbar.update(1)

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_H5 = PROCESSED_DIR / f"ml_dataset_part_{args.node_id}.h5"

    if not METADATA_FILE.exists():
        print("Error: Metadata file missing.")
        exit()
        
    loaded = np.load(METADATA_FILE, allow_pickle=True)
    metadata = loaded["data"].item() if 'data' in loaded else loaded
    
    meta_lookup = {
        pid.lower(): (ccds, smis) 
        for pid, ccds, smis in zip(metadata["pdb_ids"], metadata["ligand_names"], metadata["ligand_smiles"])
    }

    print(f"Node {args.node_id}/{args.total_nodes}: Initializing...")
    
    existing_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    chunk_size = math.ceil(len(existing_dirs) / args.total_nodes)
    start_idx = args.node_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(existing_dirs))
    
    my_dirs = existing_dirs[start_idx:end_idx]
    random.shuffle(my_dirs)
    
    print(f"Node {args.node_id}: Assigned directories {start_idx} to {end_idx} (Count: {len(my_dirs)})")

    valid_tasks = []
    for d in my_dirs:
        if TEST_LIMIT is not None and len(valid_tasks) >= TEST_LIMIT: break
        pdb_id = d.name.lower()
        if pdb_id not in meta_lookup: continue
        struct_files = list(d.glob("*.cif"))
        if not struct_files: struct_files = list(d.glob("*.pdb"))
        map_files = list(d.glob("*.map.gz"))
        
        if struct_files and map_files:
            possible_ccds, possible_smiles = meta_lookup[pdb_id]
            for ccd, smi in zip(possible_ccds, possible_smiles):
                if ONLY_TEST_AMINO_ACIDS and ccd not in AMINO_ACIDS: continue 
                if ccd in IGNORED_LIGANDS: continue
                valid_tasks.append((pdb_id, struct_files[0], map_files[0], ccd, smi))
                if TEST_LIMIT is not None and len(valid_tasks) >= TEST_LIMIT: break 

    print(f"Node {args.node_id}: Found {len(valid_tasks)} tasks.")
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

        print(f"Node {args.node_id}: Starting Processing...")

        with tqdm(total=len(valid_tasks), desc=f"Node {args.node_id}") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_task, task, h5, pbar) for task in valid_tasks]
                for future in futures:
                    future.result()

        final_count = global_write_index
        print(f"\nNode {args.node_id}: Complete. Saved {final_count} instances.")
        for key in h5.keys():
            h5[key].resize(final_count, axis=0)

    # ----------------------------
    # STATS GENERATION
    # ----------------------------
    print(f"Node {args.node_id}: Generating statistics files...")
    
    # 1. Create DataFrame from processing stats
    stats_list = []
    for pid, data in processing_stats.items():
        stats_list.append({
            "PDB_ID": pid,
            "Saved_Count": data['saved_count'],
            "Removed_Count": data['removed_count'],
            "Saved_Ligands": ", ".join(data['saved_names']),
            "Removed_Ligands": ", ".join(data['removed_names'])
        })
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_excel(STATS_EXCEL, index=False)
    print(f"Saved detailed stats to {STATS_EXCEL}")

    # 2. Generate Distribution Plot
    if saved_ligand_counter:
        plt.figure(figsize=(12, 6))
        
        # Sort by count and take top 30
        sorted_counts = sorted(saved_ligand_counter.items(), key=lambda x: x[1], reverse=True)
        top_n = min(30, len(sorted_counts))
        top_counts = sorted_counts[:top_n]
        
        labels, values = zip(*top_counts)
        
        sns.barplot(x=list(labels), y=list(values), palette='viridis')
        plt.title(f"Top {top_n} Saved Ligands (Node {args.node_id})")
        plt.xlabel("Ligand Name")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(DISTRIBUTION_PLOT)
        print(f"Saved distribution plot to {DISTRIBUTION_PLOT}")
    else:
        print("No ligands saved, skipping plot generation.")

    print(f"Dataset part saved: {OUTPUT_H5}")