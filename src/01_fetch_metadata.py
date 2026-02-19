import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- CHEMICAL INTELLIGENCE ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("WARNING: RDKit not found. Classification will be limited.")
    RDKIT_AVAILABLE = False

# ----------------------------
# CONFIGURATION
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_XLSX = DATA_DIR / "pdb_em_metadata_balanced.xlsx"
OUTPUT_NPZ = DATA_DIR / "pdb_em_metadata_balanced.npz"

# Plots
PIE_CHART_BEFORE = DATA_DIR / "ligand_class_distribution_BEFORE.png"
PIE_CHART_AFTER = DATA_DIR / "ligand_class_distribution_AFTER.png"
LIGAND_PLOT_BEFORE = DATA_DIR / "ligand_diversity_plot.png"
LIGAND_PLOT_AFTER = DATA_DIR / "ligand_diversity_balanced.png"
STATS_BEFORE = DATA_DIR / "ligand_stats.csv"
STATS_AFTER = DATA_DIR / "ligand_stats_balanced.csv"
OLIGO_PLOT_ALL = DATA_DIR / "oligomer_state_distribution_ALL_FETCHED.png"

INPUT_PARQUET = "/mnt/cephfs/projects/2023040300_LGIC_under_voltage/PLINDER_2024-06/v2/index/annotation_table.parquet"
MAX_WORKERS = 20
RESOLUTION_CUTOFF = 4.0
MAX_OLIGOMERIC_STATE = 10 

# BALANCING CONFIGURATION
TARGET_SAMPLES_PER_LIGAND = 50   # We want exactly 50 valid ones
OVERSAMPLE_FACTOR = 3            # Fetch 150 to ensure we find 50 good ones after filtering

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# LISTS (Excluded items)
# ----------------------------
IONS = {
    'ZN', 'MG', 'CA', 'MN', 'FE', 'FE2', 'NI', 'CU', 'CO', 
    'NA', 'K', 'LI', 'CL', 'IOD', 'BR', 'SR', 'CD', 'HG', 
    'SO4', 'PO4', 'NO3', 'AZI', 'NH4'
}
LIPIDS_DETERGENTS = {
    'CHL', 'CLR', 'OLA', 'PLM', 'STE', 'MYR', 'PAL', 'POP', 'POPC', 'POPE', 'DMPC', 'DPPC', 'LPC',
    'DDM', 'BOG', 'LMT', 'DM', 'NG3', 'UDM', 'LDA', 'A85', 'DET', 'UNL'
}
SUGARS = {'NAG', 'NDG', 'MAN', 'BMA', 'GAL', 'GLA', 'FUC', 'SIA', 'FUL', 'XYP', 'GLC', 'SUC', 'TRE', 'NGA', 'AHR'}
COFACTORS = {'HEM', 'HEA', 'FAD', 'FMN', 'NAD', 'NAP', 'ADP', 'ATP', 'GTP', 'GDP', 'AMP', 'SAM', 'SAH', 'TPP', 'PLP'}
AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEP', 'TPO', 'PTR', 'PCA'
}
BUFFERS_SOLVENTS = {'HOH', 'DOD', 'EDO', 'GOL', 'PEG', 'PG4', 'PGE', 'MES', 'HEPES', 'TRIS', 'EPE', 'ACY', 'FMT', 'ACT'}
EXCLUDE_LIST = IONS | BUFFERS_SOLVENTS | SUGARS | LIPIDS_DETERGENTS

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def classify_ligand_smart(row):
    code = row['ligand_ccd_code'].upper()
    smiles = row.get('ligand_smiles', '')
    qed = row.get('ligand_qed', 0.0)
    
    if code in AMINO_ACIDS: return "Amino Acids"
    if code in SUGARS: return "Saccharides"
    if code in COFACTORS: return "Cofactors/Nucleotides"
    # Note: We check the explicit list first, but RDKit below will catch the rest
    if code in LIPIDS_DETERGENTS: return "Lipids/Detergents"
    if code in IONS: return "Ions (Excluded)"
    if code in BUFFERS_SOLVENTS: return "Buffers/Solvents (Excluded)"

    if RDKIT_AVAILABLE and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                # Heuristic: Long flexible chains are usually lipids/detergents
                if rot_bonds > 12 and mw > 250: return "Lipids/Detergents"
                if mw < 300: return "Fragments/Metabolites"
        except: pass 

    if qed >= 0.4: return "Drug-like"
    return "Other"

def plot_class_pie_chart(df, plot_path, title):
    if 'assigned_class' not in df.columns:
        df['assigned_class'] = df.apply(classify_ligand_smart, axis=1)
    counts = df['assigned_class'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def plot_ligand_distribution(df, output_path, title, top_n=20):
    counts = df['ligand_ccd_code'].value_counts().head(top_n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_statistics(df, output_path):
    stats = df['ligand_ccd_code'].value_counts().reset_index()
    stats.columns = ['Ligand', 'Count']
    stats.to_csv(output_path, index=False)

def plot_oligomer_distribution(oligomer_list, output_path):
    # Visualize EVERYTHING we fetched
    valid_oligs = [float(x) for x in oligomer_list if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit()]
    plt.figure(figsize=(10, 6))
    plt.hist(valid_oligs, bins=range(1, 30), color='salmon', edgecolor='black', align='left')
    plt.title("Oligomeric State Distribution (All Fetched Candidates)")
    plt.xlabel("Oligomeric State")
    plt.ylabel("Frequency")
    plt.axvline(x=MAX_OLIGOMERIC_STATE + 0.5, color='red', linestyle='--', linewidth=2, label=f'Cutoff ({MAX_OLIGOMERIC_STATE})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def get_pdbs_by_resolution_and_method(max_res=4.0, method="ELECTRON MICROSCOPY"):
    print(f" Querying RCSB for {method} < {max_res}Å...")
    query = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less", "value": max_res}},
                {"type": "terminal", "service": "text", "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": method}}
            ]
        },
        "return_type": "entry", "request_options": {"return_all_hits": True}
    }
    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=30)
        r.raise_for_status()
        return set([x['identifier'].lower() for x in r.json().get('result_set', [])])
    except: return set()

def fetch_pdb_metadata(pdb_id):
    try:
        r_entry = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=10)
        res, olig = "N/A", "N/A"
        emdb = []
        if r_entry.ok:
            d = r_entry.json()
            res = d.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]
            if not res: res = d.get('em_3d_reconstruction', [{}])[0].get('resolution', "N/A")
            emdb = d.get("rcsb_entry_container_identifiers", {}).get("emdb_ids", [])
        
        r_assembly = requests.get(f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/1", timeout=10)
        if r_assembly.ok:
            a_data = r_assembly.json()
            olig = a_data.get('rcsb_assembly_info', {}).get('polymer_entity_instance_count_protein', "N/A")
        return pdb_id, res, olig, emdb
    except: return pdb_id, "Error", "Error", []

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not Path(INPUT_PARQUET).exists(): return

    print(f" Reading Parquet...")
    df = pd.read_parquet(INPUT_PARQUET, engine='pyarrow')
    df['entry_pdb_id_norm'] = df['entry_pdb_id'].str.lower()
    
    valid_pdb_set = get_pdbs_by_resolution_and_method(max_res=RESOLUTION_CUTOFF)
    
    # 1. LOAD EVERYTHING (Initial rough filter)
    clean_df = df[
        (df['entry_determination_method'] == 'ELECTRON MICROSCOPY') &
        (df['ligand_is_oligo']==False) &
        (~df['ligand_ccd_code'].isin(EXCLUDE_LIST)) & 
        (df['entry_pdb_id_norm'].isin(valid_pdb_set)) 
    ].copy()

    print(" Classifying ligands (RDKit Smart Check)...")
    clean_df['assigned_class'] = clean_df.apply(classify_ligand_smart, axis=1)

    # ---------------------------------------------------------
    # CRITICAL FIX: Filter based on the 'Smart' Classification
    # This removes anything RDKit identified as a lipid, even if 
    # it wasn't in our hardcoded exclusion list.
    # ---------------------------------------------------------
    initial_count = len(clean_df)
    clean_df = clean_df[clean_df['assigned_class'] != 'Lipids/Detergents']
    removed_count = initial_count - len(clean_df)
    print(f" [FILTER] Removed {removed_count} additional lipids detected by RDKit logic.")
    # ---------------------------------------------------------

    # --- PLOTS BEFORE BALANCING ---
    print(" Generating 'Before' plots...")
    plot_class_pie_chart(clean_df.sample(min(10000, len(clean_df))), PIE_CHART_BEFORE, "Before Balancing")
    plot_ligand_distribution(clean_df, LIGAND_PLOT_BEFORE, "Top 20 Ligands (Before)")
    save_statistics(clean_df, STATS_BEFORE)

    # --- OVERSAMPLING ---
    print(f" Oversampling: Selecting {TARGET_SAMPLES_PER_LIGAND * OVERSAMPLE_FACTOR} candidates per ligand...")
    
    unique_combinations = clean_df.drop_duplicates(subset=['ligand_ccd_code', 'entry_pdb_id_norm'])
    
    oversampled_df = unique_combinations.groupby('ligand_ccd_code').apply(
        lambda x: x.sample(n=min(len(x), TARGET_SAMPLES_PER_LIGAND * OVERSAMPLE_FACTOR), random_state=42)
    ).reset_index(drop=True)

    pdbs_to_fetch = list(oversampled_df['entry_pdb_id_norm'].unique())
    
    print(f" Fetching metadata for {len(pdbs_to_fetch)} unique structures (Oversampled Pool)...")
    
    resolutions, emdb_map, olig_states = {}, {}, {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_pdb_metadata, p): p for p in pdbs_to_fetch}
        for f in tqdm(as_completed(futures), total=len(futures)):
            pid, res, olig, emdb = f.result()
            resolutions[pid] = res
            olig_states[pid] = olig
            emdb_map[pid] = emdb

    # --- PLOT OLIGOMERS (ALL) ---
    print(" Generating Oligomer Distribution plot (All Fetched Data)...")
    all_values = list(olig_states.values())
    plot_oligomer_distribution(all_values, OLIGO_PLOT_ALL)

    # --- FILTERING & BALANCING ---
    print(f" Filtering & Capping to max {TARGET_SAMPLES_PER_LIGAND} valid structures per ligand...")
    
    final_rows = []
    ligand_counters = Counter()
    
    for _, row in tqdm(oversampled_df.iterrows(), total=len(oversampled_df)):
        ligand = row['ligand_ccd_code']
        pdb_id = row['entry_pdb_id_norm']
        
        if ligand_counters[ligand] >= TARGET_SAMPLES_PER_LIGAND:
            continue
            
        # CHECK Oligomer State
        olig = olig_states.get(pdb_id, "N/A")
        
        if isinstance(olig, (int, float)):
            if olig > MAX_OLIGOMERIC_STATE:
                continue 
        elif olig == "N/A":
             pass 

        final_rows.append({
            "PDB_ID": pdb_id,
            "Ligand_Names": ligand, 
            "SMILES": row.get('ligand_smiles', ''),
            "Class": row.get('assigned_class', 'Other'),
            "Resolution": resolutions.get(pdb_id, "N/A"),
            "Oligomeric_State": olig,
            "EMDB_IDs": emdb_map.get(pdb_id, [])
        })
        ligand_counters[ligand] += 1

    # --- PLOTS AFTER BALANCING ---
    final_df_flat = pd.DataFrame(final_rows)
    
    print(" Generating 'After' plots (Final Valid Dataset)...")
    plot_class_pie_chart(final_df_flat.rename(columns={'Class': 'assigned_class'}), PIE_CHART_AFTER, "After Balancing")
    
    counts = final_df_flat['Ligand_Names'].value_counts().reset_index()
    counts.columns = ['ligand_ccd_code', 'count']
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts['ligand_ccd_code'].head(20), y=counts['count'].head(20), palette='viridis')
    plt.title("Top 20 Ligands (After Balancing)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(LIGAND_PLOT_AFTER)
    counts.to_csv(STATS_AFTER, index=False)

    # --- SAVE FILES ---
    grouped_final = final_df_flat.groupby('PDB_ID')
    
    pdb_out, lig_out, smiles_out, res_out, olig_out, emdb_out = [], [], [], [], [], []
    excel_data = []

    for pid, group in grouped_final:
        unique_ligs = list(set(group['Ligand_Names']))
        unique_smiles = list(set(group['SMILES']))
        unique_classes = list(set(group['Class']))
        res = group.iloc[0]['Resolution']
        olig = group.iloc[0]['Oligomeric_State']
        emdb = group.iloc[0]['EMDB_IDs']

        pdb_out.append(pid)
        lig_out.append(unique_ligs)
        smiles_out.append(unique_smiles)
        res_out.append(res)
        olig_out.append(olig)
        emdb_out.append(emdb)

        excel_data.append({
            "PDB_ID": pid,
            "Ligand_Names": ",".join(unique_ligs),
            "Ligand_Classes": ",".join(unique_classes),
            "SMILES": " | ".join(unique_smiles),
            "Resolution": res,
            "Oligomeric_State": olig,
            "EMDB_IDs": ",".join(emdb) if isinstance(emdb, list) else str(emdb)
        })

    print(f" Saving {len(pdb_out)} unique structures to {OUTPUT_NPZ}...")
    np.savez(OUTPUT_NPZ, data={
        "pdb_ids": np.array(pdb_out),
        "ligand_names": np.array(lig_out, dtype=object),
        "ligand_smiles": np.array(smiles_out, dtype=object),
        "resolutions": np.array(res_out, dtype=object),
        "olig_states": np.array(olig_out, dtype=object),
        "emdb_ids": np.array(emdb_out, dtype=object)
    })

    pd.DataFrame(excel_data).to_excel(OUTPUT_XLSX, index=False)
    print(f" Saved metadata to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()