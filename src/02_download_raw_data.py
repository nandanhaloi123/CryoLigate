import numpy as np
import requests
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Adjust these paths if necessary to match your exact folder structure
INPUT_NPZ = PROJECT_ROOT / "data" / "metadata" / "pdb_em_metadata_balanced.npz"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = PROJECT_ROOT / "data" / "download.log"

MAX_WORKERS = 20

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# ----------------------------
# DOWNLOAD FUNCTION
# ----------------------------
def download_file(url, out_path):
    """
    Downloads a file. Returns status code (0=Skipped, 1=Success, 2=Fail)
    """
    out_path = Path(out_path)
    
    # 1. Check if exists
    if out_path.exists():
        logging.info(f"SKIPPED_EXISTS: {out_path.name}")
        return 0
    
    # 2. Try Download
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        logging.info(f"SUCCESS: {out_path.name}")
        return 1
    
    # 3. Handle Errors
    except Exception as e:
        if r.status_code == 404:
            logging.warning(f"NOT_FOUND_404: {out_path.name}")
        else:
            logging.error(f"FAILED: {out_path.name} from {url} | Error: {str(e)}")
        return 2

# ----------------------------
# MAIN
# ----------------------------
def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_NPZ.exists():
        print(f"Error: Metadata file not found at {INPUT_NPZ}")
        return

    print(f"Loading metadata from {INPUT_NPZ}...")
    
    # --- ROBUST LOADING BLOCK ---
    try:
        loaded = np.load(INPUT_NPZ, allow_pickle=True)
        
        # Check if the data is wrapped in a 'data' dictionary key (as per previous script)
        if "data" in loaded:
            data = loaded["data"].item()
        else:
            # Or if it was saved as individual arrays
            data = {k: loaded[k] for k in loaded.files}
            
        # DEBUG: Print keys to ensure we are right
        print(f"Keys found in NPZ: {list(data.keys())}")

        # Correct Mapping (Lowercase keys)
        pdb_ids = data["pdb_ids"]
        emdb_ids_list = data["emdb_ids"]
        # olig_states = data["olig_states"] # Available if needed

    except KeyError as e:
        print(f"\nCRITICAL ERROR: The key {e} was not found in the .npz file.")
        print("Please check the 'Keys found' list above.")
        return
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return
    # ----------------------------

    futures = []
    queued_count = 0

    print(f"Starting downloads for {len(pdb_ids)} structures...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # Iterate through all entries
        for i, pdb_id in enumerate(pdb_ids):
            
            queued_count += 1
            entry_dir = DOWNLOAD_DIR / pdb_id.lower()
            entry_dir.mkdir(exist_ok=True)
            
            # 1. PDB Download
            pdb_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
            pdb_out = entry_dir / f"{pdb_id.lower()}.pdb"
            futures.append(ex.submit(download_file, pdb_url, pdb_out))
            
            # 2. EMDB Download
            emdbs = emdb_ids_list[i]
            
            # Handle various formats (list, array, comma-string)
            if isinstance(emdbs, str):
                emdbs = emdbs.split(",") if emdbs else []
            elif isinstance(emdbs, (np.ndarray, list)):
                pass # Already a list
            else:
                emdbs = []

            for emdb in emdbs:
                emdb = str(emdb).strip()
                if not emdb: continue
                
                # Clean ID (e.g., EMD-1234 -> 1234)
                num = emdb.replace("EMD-", "")
                
                # Using .map.gz
                emdb_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb}/map/emd_{num}.map.gz"
                emdb_out = entry_dir / f"{emdb}.map.gz"
                futures.append(ex.submit(download_file, emdb_url, emdb_out))

        print(f" -> Queued {queued_count} structures.")

        # --- Monitor Progress ---
        stats = {0: 0, 1: 0, 2: 0} 
        
        if futures:
            for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
                result_code = f.result()
                stats[result_code] += 1
        else:
            print("No files to download.")

    # --- Final Summary ---
    print("\n" + "="*40)
    print(f"DOWNLOAD SUMMARY")
    print("="*40)
    print(f"✅ Downloaded: {stats[1]}")
    print(f"⏭️  Skipped:    {stats[0]} (Already exists)")
    print(f"❌ Failed:     {stats[2]}")
    print("-" * 40)
    
    if stats[2] > 0:
        print(f"⚠️  Check {LOG_FILE} to see which files failed.")
    else:
        print(f"🎉 All files processed successfully.")

if __name__ == "__main__":
    main()