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
RESOLUTION_CUTOFF = 4.0

DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- EXPANDED EXCLUDE LIST ---
# Grouped for clarity. You can add more as you find them.
IONS = [
    'ZN', 'MG', 'CA', 'MN', 'FE', 'FE2', 'NI', 'CU', 'CO', # Metals
    'NA', 'K', 'LI', 'CL', 'IOD', 'BR', 'SR', 'CD', 'HG', # Salts/Heavy atoms
    'SO4', 'PO4', 'NO3', 'AZI' # Common Ion clusters
]
SOLVENTS_BUFFERS = [
    'HOH', 'DOD', 'WAU', # Water
    'GOL', 'EDO', 'PG4', 'PEG', 'PGE', 'PE4', '1PE', # PEG/Glycerol series
    'DMS', 'EOH', 'MOH', 'ACY', 'ACE', 'FMT', 'CIT', 'MES', 'TRS', 'BCT', 'IPA' # Solvents/Buffers
]
GLYCANS = [
    'NAG', 'NDG', 'MAN', 'BMA', 'GAL', 'GLA', 'FUC', 'SIA', 'FUL', 'XYP'
]
LIPIDS_DETERGENTS = [
    # Lipids
    # 'LPC', 'CLM', 'CHL', 'CLR', 'OLA', 'PLM', 'STE', 'MYR', 'PAL',
    # 'POP', 'POPC', 'POPE', 'DMPC', 'DPPC', 
    # Detergents (Common in Cryo-EM)
    'BOG', 'DDM', 'DM', 'LMT', 'HTG', 'Y01', 'LDA', 'UNL'
]

EXCLUDE_LIST = set(IONS + SOLVENTS_BUFFERS + GLYCANS + LIPIDS_DETERGENTS)


def get_pdbs_by_resolution_and_method(max_res=4.0, method="ELECTRON MICROSCOPY"):
    """
    Queries RCSB Search API to get a list of PDB IDs matching criteria.
    This is much faster than downloading metadata for every entry.
    """
    print(f" Querying RCSB Search API for {method} structures < {max_res}Å...")
    
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": max_res
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": method
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True
        }
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    try:
        r = requests.post(url, json=query, timeout=30)
        r.raise_for_status()
        data = r.json()
        pdb_ids = [result['identifier'].lower() for result in data.get('result_set', [])]
        print(f" Found {len(pdb_ids)} PDBs matching criteria.")
        return set(pdb_ids)
    except Exception as e:
        print(f" Error querying RCSB Search API: {e}")
        return set()


def fetch_pdb_metadata(pdb_id):
    """
    Fetches exact resolution and EMDB IDs for saving.
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            data = r.json()
            # Try combined resolution first, fallback to EM resolution
            resolution = data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]
            if not resolution:
                 resolution = data.get('em_3d_reconstruction', [{}])[0].get('resolution', "N/A")
            
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

    # 1. READ PARQUET
    print(f" Reading Parquet from: {input_path}")
    df = pd.read_parquet(input_path, engine='pyarrow')
    print(f" Initial entries: {len(df)}")

    # 2. FETCH VALID PDB LIST FROM RCSB DIRECTLY
    # We do this first so we don't waste time processing rows we will delete later.
    valid_pdb_set = get_pdbs_by_resolution_and_method(max_res=RESOLUTION_CUTOFF)

    if not valid_pdb_set:
        print("Warning: API returned no PDBs or failed. Proceeding with caution (filtering might fail).")

    # 3. APPLY FILTERS
    # Convert PDB ID column to lower case for comparison
    df['entry_pdb_id_norm'] = df['entry_pdb_id'].str.lower()
    
    filtered_df = df[
        (~df['ligand_is_covalent']) &
        (df['entry_determination_method'] == 'ELECTRON MICROSCOPY') &
        (df['ligand_is_oligo']==False) &
        (df['ligand_is_other']==False) &
        # Exclude artifacts
        (~df['ligand_ccd_code'].isin(EXCLUDE_LIST)) & 
        # Filter by Resolution List from RCSB
        (df['entry_pdb_id_norm'].isin(valid_pdb_set)) 
    ].copy()

    print(f" Filtered to {len(filtered_df)} entries (EM, <{RESOLUTION_CUTOFF}Å, Non-cov, Cleaned Ligands).")
    
    grouped = filtered_df.groupby('entry_pdb_id_norm')
    pdb_ids_list, lig_names, lig_smiles = [], [], []

    for pdb, group in grouped:
        pdb_ids_list.append(pdb)
        # Use set to avoid duplicate ligand names per pdb in the list
        lig_names.append(list(set(group['ligand_ccd_code'].tolist())))
        lig_smiles.append(list(set(group['ligand_smiles'].tolist())))

    print(f" Querying specific metadata for {len(pdb_ids_list)} unique structures...")
    
    resolutions, emdb_map = {}, {}
    
    # 4. FETCH SPECIFIC VALUES (THREAD POOL)
    # Even though we filtered by resolution, we still fetch the specific number for your Excel sheet
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_pdb_metadata, pid): pid for pid in pdb_ids_list}
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Fetching Metadata", unit="pdb"):
            pid, res, emdb = f.result()
            resolutions[pid], emdb_map[pid] = res, emdb

    # 5. SAVE NPZ
    data_dict = {
        "pdb_ids": np.array(pdb_ids_list),
        "ligand_names": np.array(lig_names, dtype=object),
        "ligand_smiles": np.array(lig_smiles, dtype=object),
        "resolutions": np.array([resolutions.get(p) for p in pdb_ids_list], dtype=object),
        "emdb_ids": np.array([emdb_map.get(p, []) for p in pdb_ids_list], dtype=object)
    }
    np.savez(OUTPUT_NPZ, data=data_dict)
    print(f"\n Saved Metadata NPZ to: {OUTPUT_NPZ}")

    # 6. SAVE EXCEL
    print(" Saving Excel summary...")
    excel_data = []
    for i, pid in enumerate(pdb_ids_list):
        excel_data.append({
            "PDB_ID": pid,
            "Ligand_Names": ",".join(data_dict["ligand_names"][i]),
            "SMILES": " | ".join(data_dict["ligand_smiles"][i]), # Pipe separator for readability
            "Resolution": data_dict["resolutions"][i],
            "EMDB_IDs": ",".join(data_dict["emdb_ids"][i])
        })
    
    df_excel = pd.DataFrame(excel_data)
    df_excel.to_excel(OUTPUT_XLSX, index=False)
    print(f" Saved Metadata XLSX to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()