import numpy as np
import requests
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_NPZ = PROJECT_ROOT / "data" / "metadata" / "pdb_em_metadata_balanced.npz"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = PROJECT_ROOT / "data" / "download.log"

MAX_WORKERS = 20
OLIGOMER_CUTOFF = 10  # Only download if state < 10

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

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
        logging.error(f"FAILED: {out_path.name} from {url} | Error: {str(e)}")
        return 2

def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_NPZ.exists():
        print(f"Error: Metadata file not found at {INPUT_NPZ}")
        return

    print(f"Loading metadata from {INPUT_NPZ}...")
    data = np.load(INPUT_NPZ, allow_pickle=True)["data"].item()
    
    pdb_ids = data["pdb_ids"]
    emdb_ids_list = data["emdb_ids"]
    olig_states = data["olig_states"]  # Retrieve oligomer states

    futures = []
    skipped_count = 0
    queued_count = 0

    print(f"Starting downloads (Filter: Oligomeric State < {OLIGOMER_CUTOFF})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # Zip all arrays to iterate together
        for i, pdb_id in enumerate(pdb_ids):
            state_raw = olig_states[i]
            
            # --- FILTERING LOGIC ---
            try:
                # Try to convert to int (handles strings like '5' or floats like 5.0)
                state_val = int(state_raw)
                
                if state_val >= OLIGOMER_CUTOFF:
                    logging.info(f"SKIPPED_FILTER: {pdb_id} (Oligomeric State {state_val} >= {OLIGOMER_CUTOFF})")
                    skipped_count += 1
                    continue
            except (ValueError, TypeError):
                # If state is "N/A", "Error", or None, we skip it to be safe
                logging.warning(f"SKIPPED_INVALID: {pdb_id} (Invalid State: {state_raw})")
                skipped_count += 1
                continue
            # -----------------------

            queued_count += 1
            emdbs = emdb_ids_list[i]
            entry_dir = DOWNLOAD_DIR / pdb_id.lower()
            entry_dir.mkdir(exist_ok=True)
            
            # Submit PDB
            pdb_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
            pdb_out = entry_dir / f"{pdb_id.lower()}.pdb"
            futures.append(ex.submit(download_file, pdb_url, pdb_out))
            
            # Submit EMDBs
            if isinstance(emdbs, (list, np.ndarray)):
                for emdb in emdbs:
                    # Clean ID (e.g., EMD-1234 -> 1234)
                    num = emdb.replace("EMD-", "")
                    emdb_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb}/map/emd_{num}.map.gz"
                    emdb_out = entry_dir / f"{emdb}.map.gz"
                    futures.append(ex.submit(download_file, emdb_url, emdb_out))

        print(f" -> Queued {queued_count} structures.")
        print(f" -> Skipped {skipped_count} structures (Oligomer >= {OLIGOMER_CUTOFF} or Invalid).")

        # --- Monitor Progress ---
        stats = {0: 0, 1: 0, 2: 0} 
        
        if futures:
            for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
                result_code = f.result()
                stats[result_code] += 1
        else:
            print("No files to download matching criteria.")

    # --- Final Summary ---
    print("\n" + "="*40)
    print(f"DOWNLOAD SUMMARY")
    print("="*40)
    print(f"✅ Downloaded: {stats[1]}")
    print(f"⏭️  Skipped:    {stats[0]} (Already existed)")
    print(f"❌ Failed:     {stats[2]}")
    print("-" * 40)
    
    if stats[2] > 0:
        print(f"⚠️  Check {LOG_FILE} to see which files failed.")
    else:
        print(f"🎉 All processed successfully.")

if __name__ == "__main__":
    main()