import h5py
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
DEFAULT_PROCESSED_DIR = Path("data/processed")

def parse_args():
    parser = argparse.ArgumentParser(description="Verify consistency for ALL HDF5 files.")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_PROCESSED_DIR), help="Path to the processed data directory")
    return parser.parse_args()

def verify_single_h5(h5_path, data_dir):
    """Verifies a single H5 file against the file system."""
    print(f"\n--- Checking: {h5_path.name} ---")
    
    missing_map = 0
    missing_pdb = 0
    missing_dir = 0
    errors = []
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'pdb_ids' not in f or 'ligand_names' not in f:
                print(f"   [SKIP] {h5_path.name} is missing required datasets.")
                return 0, []

            pdb_ids = f['pdb_ids']
            ligand_names = f['ligand_names']
            total = len(pdb_ids)
            
            # Use tqdm for progress bar
            for i in tqdm(range(total), desc=f"Scanning {h5_path.name}", unit="entry", leave=False):
                pdb_id = pdb_ids[i].decode('utf-8').lower()
                lig_name = ligand_names[i].decode('utf-8') # e.g. "ATP" or "ATP_1"
                
                # 1. Check Directory
                pdb_dir = data_dir / pdb_id
                if not pdb_dir.exists():
                    errors.append(f"[{h5_path.name}] Idx {i}: Missing Dir -> {pdb_id}")
                    missing_dir += 1
                    continue 

                # 2. Check Map File
                # Look for exact match or pattern match
                map_path_exact = pdb_dir / f"ml_map_{lig_name}.mrc"
                map_found = map_path_exact.exists()
                
                if not map_found:
                    # Fallback check for generic naming if exact fails
                    if list(pdb_dir.glob(f"ml_map_{lig_name}_*.mrc")):
                        map_found = True
                
                if not map_found:
                    errors.append(f"[{h5_path.name}] Idx {i}: Missing MAP -> {pdb_id}/{lig_name}")
                    missing_map += 1

                # 3. Check PDB File
                pdb_path_exact = pdb_dir / f"{pdb_id}_{lig_name}_clean.pdb"
                pdb_found = pdb_path_exact.exists()
                
                if not pdb_found:
                    # Fallback check
                    if list(pdb_dir.glob(f"{pdb_id}_{lig_name}_*_clean.pdb")):
                        pdb_found = True
                
                if not pdb_found:
                    errors.append(f"[{h5_path.name}] Idx {i}: Missing PDB -> {pdb_id}/{lig_name}")
                    missing_pdb += 1
                    
            return total, errors

    except Exception as e:
        print(f"   [ERROR] Could not read {h5_path.name}: {e}")
        return 0, [f"CRITICAL: {e}"]

def main():
    args = parse_args()
    data_dir = Path(args.dir)
    
    if not data_dir.exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        return

    # Find all H5 files
    h5_files = sorted(list(data_dir.glob("*.h5")))
    
    if not h5_files:
        print(f"❌ No .h5 files found in {data_dir}")
        return

    print(f"Found {len(h5_files)} HDF5 files to check.\n")
    
    total_entries_all = 0
    all_errors = []
    
    for h5_file in h5_files:
        count, errors = verify_single_h5(h5_file, data_dir)
        total_entries_all += count
        all_errors.extend(errors)

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("           GLOBAL VERIFICATION REPORT           ")
    print("="*50)
    print(f"Files Checked: {len(h5_files)}")
    print(f"Total Entries: {total_entries_all}")
    print(f"Total Errors:  {len(all_errors)}")
    print("-" * 50)

    if len(all_errors) == 0:
        print("✅ SUCCESS: All datasets are consistent!")
    else:
        print(f"❌ FAILURE: Found {len(all_errors)} inconsistencies across all files.")
        print("\n--- First 20 Errors ---")
        for err in all_errors[:20]:
            print(err)
        if len(all_errors) > 20:
            print(f"... and {len(all_errors) - 20} more.")
            
        # Optional: Save error log
        log_path = data_dir / "verification_errors.log"
        with open(log_path, "w") as f:
            for err in all_errors:
                f.write(err + "\n")
        print(f"\nFull error log saved to: {log_path}")

if __name__ == "__main__":
    main()