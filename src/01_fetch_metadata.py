import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_XLSX = DATA_DIR / "pdb_em_metadata.xlsx"
OUTPUT_NPZ = DATA_DIR / "pdb_em_metadata.npz"

INPUT_PARQUET = "/mnt/cephfs/projects/2023040300_LGIC_under_voltage/PLINDER_2024-06/v2/index/annotation_table.parquet"
MAX_WORKERS = 20

DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_pdb_info(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            data = r.json()
            resolution = data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]
            resolution = resolution if resolution else "N/A"
            emdb_ids = data.get("rcsb_entry_container_identifiers", {}).get("emdb_ids", [])
        else:
            resolution, emdb_ids = "Error", []
    except Exception as e:
        resolution, emdb_ids = str(e), []
    return pdb_id, resolution, emdb_ids


def main():
    input_path = Path(INPUT_PARQUET)
    
    if not input_path.exists():
        print(f"Error: Parquet file not found at {input_path}")
        return

    print(f" Reading Parquet from: {input_path}")
    df = pd.read_parquet(input_path, engine='pyarrow')
    
    exclude = ['NAG', 'LPC', 'BMA', 'MAN', 'ZN', 'MG', 'HOH', 'SO4']
    filtered_df = df[
        (~df['ligand_is_covalent']) &
        (df['entry_oligomeric_state'] == 'monomeric') &
        (df['ligand_qed'] >= 0.7) &
        (df['entry_determination_method'] == 'ELECTRON MICROSCOPY') &
        (~df['ligand_ccd_code'].isin(exclude))
    ].copy()

    print(f" Filtered to {len(filtered_df)} entries.")
    
    grouped = filtered_df.groupby(filtered_df['entry_pdb_id'].str.lower())
    pdb_ids, lig_names, lig_smiles = [], [], []

    for pdb, group in grouped:
        pdb_ids.append(pdb)
        lig_names.append(group['ligand_ccd_code'].tolist())
        lig_smiles.append(group['ligand_smiles'].tolist())

    print(f" Querying RCSB for {len(pdb_ids)} structures...")
    
    resolutions, emdb_map = {}, {}
    
    # --- THREAD POOL WITH PROGRESS BAR ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_pdb_info, pid): pid for pid in pdb_ids}
        
        # Wrap the iterator with tqdm for the progress bar
        for f in tqdm(as_completed(futures), total=len(futures), desc="Fetching Metadata", unit="pdb"):
            pid, res, emdb = f.result()
            resolutions[pid], emdb_map[pid] = res, emdb

    # --- SAVE NPZ ---
    data_dict = {
        "pdb_ids": np.array(pdb_ids),
        "ligand_names": np.array(lig_names, dtype=object),
        "ligand_smiles": np.array(lig_smiles, dtype=object),
        "resolutions": np.array([resolutions.get(p) for p in pdb_ids], dtype=object),
        "emdb_ids": np.array([emdb_map.get(p, []) for p in pdb_ids], dtype=object)
    }
    np.savez(OUTPUT_NPZ, data=data_dict)
    print(f"\n Saved Metadata NPZ to: {OUTPUT_NPZ}")

    # --- SAVE EXCEL ---
    print(" Saving Excel summary...")
    excel_data = []
    for i, pid in enumerate(pdb_ids):
        excel_data.append({
            "PDB_ID": pid,
            "Ligand_Names": ",".join(data_dict["ligand_names"][i]),
            "SMILES": ",".join(data_dict["ligand_smiles"][i]),
            "Resolution": data_dict["resolutions"][i],
            "EMDB_IDs": ",".join(data_dict["emdb_ids"][i])
        })
    
    df_excel = pd.DataFrame(excel_data)
    df_excel.to_excel(OUTPUT_XLSX, index=False)
    print(f" Saved Metadata XLSX to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()