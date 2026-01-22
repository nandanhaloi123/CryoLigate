import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_XLSX = DATA_DIR / "pdb_em_metadata_balanced.xlsx"
OUTPUT_NPZ = DATA_DIR / "pdb_em_metadata_balanced.npz"
PLOT_OUTPUT = DATA_DIR / "ligand_diversity_balanced.png"
STATS_OUTPUT = DATA_DIR / "ligand_stats_balanced.csv"  # <--- NEW CSV OUTPUT

INPUT_PARQUET = "/mnt/cephfs/projects/2023040300_LGIC_under_voltage/PLINDER_2024-06/v2/index/annotation_table.parquet"
MAX_WORKERS = 20
RESOLUTION_CUTOFF = 4.0

# --- BALANCING PARAMETER ---
MAX_SAMPLES_PER_LIGAND = 50 

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
    print(f" Querying RCSB Search API for {method} structures < {max_res}Å...")
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less", "value": max_res}},
                {"type": "terminal", "service": "text", "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": method}}
            ]
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }
    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=30)
        r.raise_for_status()
        ids = set([x['identifier'].lower() for x in r.json().get('result_set', [])])
        print(f" Found {len(ids)} PDBs matching criteria.")
        return ids
    except Exception as e:
        print(f" Error: {e}")
        return set()

def fetch_pdb_metadata(pdb_id):
    try:
        r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=10)
        if r.ok:
            d = r.json()
            res = d.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]
            if not res: res = d.get('em_3d_reconstruction', [{}])[0].get('resolution', "N/A")
            return pdb_id, res, d.get("rcsb_entry_container_identifiers", {}).get("emdb_ids", [])
    except: pass
    return pdb_id, "Error", []

def plot_ligand_distribution(df, plot_path, stats_path, title_suffix=""):
    """Generates plot AND saves the raw count CSV."""
    counts = df['ligand_ccd_code'].value_counts()
    
    # 1. Save CSV
    counts.to_csv(stats_path, header=["Count"])
    print(f"   - Stats CSV saved to {stats_path}")

    # 2. Generate Plot
    plt.figure(figsize=(15, 8))
    plot_data = counts.head(50)
    bars = plt.bar(plot_data.index, plot_data.values, color='teal', edgecolor='black', alpha=0.7)
    
    plt.title(f"Ligand Diversity {title_suffix}\n(Total Unique Types: {len(counts)})", fontsize=16)
    plt.xlabel("Ligand CCD Code")
    plt.ylabel("Frequency (Count)")
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   - Plot saved to {plot_path}")

def main():
    if not Path(INPUT_PARQUET).exists(): return

    # 1. READ & FILTER
    print(f" Reading Parquet...")
    df = pd.read_parquet(INPUT_PARQUET, engine='pyarrow')
    df['entry_pdb_id_norm'] = df['entry_pdb_id'].str.lower()
    
    valid_pdb_set = get_pdbs_by_resolution_and_method(max_res=RESOLUTION_CUTOFF)
    
    clean_df = df[
        (~df['ligand_is_covalent']) &
        (df['entry_determination_method'] == 'ELECTRON MICROSCOPY') &
        (df['ligand_is_oligo']==False) &
        (df['ligand_is_other']==False) &
        (~df['ligand_ccd_code'].isin(EXCLUDE_LIST)) & 
        (df['entry_pdb_id_norm'].isin(valid_pdb_set)) 
    ].copy()

    print(f" Entries before balancing: {len(clean_df)}")
    
    # 2. APPLY DOWNSAMPLING / BALANCING
    print(f" Applying Balancing: Max {MAX_SAMPLES_PER_LIGAND} entries per ligand type...")
    
    balanced_df = clean_df.groupby('ligand_ccd_code').apply(
        lambda x: x.sample(n=min(len(x), MAX_SAMPLES_PER_LIGAND), random_state=42)
    ).reset_index(drop=True)
    
    print(f" Entries after balancing: {len(balanced_df)}")
    
    # 3. PLOT & SAVE STATS CSV
    plot_ligand_distribution(balanced_df, PLOT_OUTPUT, STATS_OUTPUT, title_suffix="(Balanced)")

    # 4. Finalize List for NPZ/Excel
    grouped = balanced_df.groupby('entry_pdb_id_norm')
    pdb_ids_list, lig_names, lig_smiles = [], [], []

    for pdb, group in grouped:
        pdb_ids_list.append(pdb)
        lig_names.append(list(set(group['ligand_ccd_code'])))
        lig_smiles.append(list(set(group['ligand_smiles'])))

    # 5. Fetch Metadata
    print(f" Fetching metadata for {len(pdb_ids_list)} structures...")
    resolutions, emdb_map = {}, {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_pdb_metadata, p): p for p in pdb_ids_list}
        for f in tqdm(as_completed(futures), total=len(futures)):
            pid, res, emdb = f.result()
            resolutions[pid], emdb_map[pid] = res, emdb

    # 6. Save Data Files
    data_dict = {
        "pdb_ids": np.array(pdb_ids_list),
        "ligand_names": np.array(lig_names, dtype=object),
        "ligand_smiles": np.array(lig_smiles, dtype=object),
        "resolutions": np.array([resolutions.get(p) for p in pdb_ids_list], dtype=object),
        "emdb_ids": np.array([emdb_map.get(p, []) for p in pdb_ids_list], dtype=object)
    }
    np.savez(OUTPUT_NPZ, data=data_dict)
    
    pd.DataFrame([{
        "PDB_ID": p,
        "Ligand_Names": ",".join(data_dict["ligand_names"][i]),
        "SMILES": " | ".join(data_dict["ligand_smiles"][i]),
        "Resolution": data_dict["resolutions"][i],
        "EMDB_IDs": ",".join(data_dict["emdb_ids"][i])
    } for i, p in enumerate(pdb_ids_list)]).to_excel(OUTPUT_XLSX, index=False)
    
    print(f" Done. Saved balanced dataset to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()