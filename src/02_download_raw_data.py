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

# Input
INPUT_NPZ = PROJECT_ROOT / "data" / "metadata" / "pdb_em_metadata_balanced.npz"

# Outputs
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_FILE = PROJECT_ROOT / "data" / "download_log.txt"

MAX_WORKERS = 10 
TIMEOUT_SECONDS = 60

# ----------------------------
# REAL-TIME LOGGING SETUP
# ----------------------------
class ForceFlushHandler(logging.FileHandler):
    """
    Custom handler that forces writing to disk immediately 
    after every single log entry.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()

# Reset logger
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

# Use the custom handler
handler = ForceFlushHandler(LOG_FILE, mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------------
# ROBUST DOWNLOAD FUNCTION
# ----------------------------
def download_file(url, out_path):
    """
    Downloads a file and logs every outcome.
    Returns: 0=Success, 1=Skipped, 2=Failed
    """
    if out_path.exists():
        if out_path.stat().st_size > 0:
            logging.info(f"SKIPPED (Exists): {out_path.name}")
            return 1 
        else:
            out_path.unlink() 

    try:
        r = requests.get(url, stream=True, timeout=TIMEOUT_SECONDS)
        
        if r.status_code == 404:
            logging.warning(f"404 Not Found: {url}")
            return 2
        
        r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logging.info(f"DOWNLOADED: {out_path.name}")
        return 0

    except Exception as e:
        logging.error(f"FAILED: {url} | Error: {str(e)}")
        if out_path.exists():
            out_path.unlink()
        return 2

# ----------------------------
# HELPER: CONSTRUCT URLS
# ----------------------------
def get_pdb_url(pdb_id):
    return f"https://files.rcsb.org/download/{pdb_id}.cif"

def get_emdb_url(emdb_id):
    # CHANGED: Switched from ftp.ebi.ac.uk to files.wwpdb.org (More stable)
    id_clean = emdb_id.replace("EMD-", "")
    return f"https://files.wwpdb.org/pub/emdb/structures/EMD-{id_clean}/map/emd_{id_clean}.map.gz"

# ----------------------------
# MAIN PROCESSING
# ----------------------------
def main():
    if not INPUT_NPZ.exists():
        print(f"Error: Input file {INPUT_NPZ} not found.")
        return

    print(f"Loading metadata from {INPUT_NPZ}...")
    try:
        npz_data = np.load(INPUT_NPZ, allow_pickle=True)
        if 'data' in npz_data:
            data = npz_data['data'].item()
        else:
            data = npz_data  
        pdb_ids = data['pdb_ids']
        emdb_ids_raw = data['emdb_ids']
    except Exception as e:
        print(f"Error reading NPZ file: {e}")
        return
    
    print(f"Found {len(pdb_ids)} structures to process.")
    print(f"Real-time logs: {LOG_FILE}")

    tasks = []
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for i, pid in enumerate(pdb_ids):
        pdb_folder = RAW_DATA_DIR / pid
        pdb_folder.mkdir(exist_ok=True)
        
        # PDB Task
        pdb_url = get_pdb_url(pid)
        pdb_path = pdb_folder / f"{pid}.cif"
        tasks.append((pdb_url, pdb_path, f"{pid}_PDB"))

        # EMDB Task
        em_list = emdb_ids_raw[i]
        eid = None
        if isinstance(em_list, (list, np.ndarray)) and len(em_list) > 0:
            eid = em_list[0] 
        elif isinstance(em_list, str) and em_list and em_list != "nan":
            eid = em_list.replace(",", "").split()[0]
            
        if eid:
            em_url = get_emdb_url(eid)
            em_path = pdb_folder / f"{eid.replace('-','_').lower()}.map.gz"
            tasks.append((em_url, em_path, f"{pid}_EMDB"))

    print(f"Queued {len(tasks)} files for download.")
    
    success_count = 0
    skip_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, url, path): (type_, path.name) for url, path, type_ in tasks}
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            type_, name = futures[f]
            try:
                result = f.result()
                if result == 0:
                    success_count += 1
                elif result == 1:
                    skip_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logging.error(f"CRITICAL EXECUTOR ERROR: {name} | {str(e)}")
                fail_count += 1

    print("\n--- Download Summary ---")
    print(f"Success: {success_count}")
    print(f"Skipped (Already Existed): {skip_count}")
    print(f"Failed: {fail_count}")
    print(f"Log file: {LOG_FILE}")

if __name__ == "__main__":
    main()