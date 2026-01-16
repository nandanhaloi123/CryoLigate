import numpy as np
import requests
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_NPZ = PROJECT_ROOT / "data" / "metadata" / "pdb_em_metadata.npz"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = PROJECT_ROOT / "data" / "download.log"

MAX_WORKERS = 20


# Configure logging to write ONLY to file, not console (to preserve progress bar)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w' # Overwrite log each run. Change to 'a' to append.
)


def download_file(url, out_path):
    """
    Downloads a file. Returns status code (0=Skipped, 1=Success, 2=Fail)
    and a message for the summary.
    """
    out_path = Path(out_path)
    filename = out_path.name
    
    # 1. Check if exists
    if out_path.exists():
        logging.info(f"SKIPPED: {filename} (Already exists)")
        return 0  # Skipped
    
    # 2. Try Download
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        logging.info(f"SUCCESS: {filename}")
        return 1  # Success
    
    # 3. Handle Errors
    except Exception as e:
        logging.error(f"FAILED: {filename} from {url} | Error: {str(e)}")
        return 2  # Failed


def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for metadata
    if not INPUT_NPZ.exists():
        print(f"Error: Metadata file not found at {INPUT_NPZ}")
        return

    print(f"Loading metadata from {INPUT_NPZ}...")
    data = np.load(INPUT_NPZ, allow_pickle=True)["data"].item()
    pdb_ids = data["pdb_ids"]
    emdb_ids_list = data["emdb_ids"]

    futures = []
    print(f"Starting downloads (Logs saved to {LOG_FILE})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for pdb_id, emdbs in zip(pdb_ids, emdb_ids_list):
            entry_dir = DOWNLOAD_DIR / pdb_id.lower()
            entry_dir.mkdir(exist_ok=True)
            
            # Submit PDB
            pdb_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
            pdb_out = entry_dir / f"{pdb_id.lower()}.pdb"
            futures.append(ex.submit(download_file, pdb_url, pdb_out))
            
            # Submit EMDBs
            if isinstance(emdbs, (list, np.ndarray)):
                for emdb in emdbs:
                    num = emdb.replace("EMD-", "")
                    emdb_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb}/map/emd_{num}.map.gz"
                    emdb_out = entry_dir / f"{emdb}.map.gz"
                    futures.append(ex.submit(download_file, emdb_url, emdb_out))

        # --- Monitor Progress & Collect Stats ---
        stats = {0: 0, 1: 0, 2: 0} # 0:Skip, 1:Success, 2:Fail
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            result_code = f.result()
            stats[result_code] += 1

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
        print(f"🎉 All files processed successfully.")

if __name__ == "__main__":
    main()