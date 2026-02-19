import h5py
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# ----------------------------
# IMPORT UTILS
# ----------------------------
try:
    # This imports the function from your util.py in the same folder
    from util import get_ligand_class_by_name
except ImportError:
    print("WARNING: Could not import 'util.py'. Make sure it is in the same directory.")
    # Fallback dummy function if util is missing
    get_ligand_class_by_name = lambda x: "Other"

# ----------------------------
# CONFIGURATION
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# Input Metadata (Source of Truth for Classes)
METADATA_FILE = METADATA_DIR / "pdb_em_metadata_balanced.xlsx"

# HDF5 Config
INPUT_H5_PATTERN = PROCESSED_DIR / "ml_dataset_part_*.h5"
OUTPUT_FINAL_H5 = PROCESSED_DIR / "ml_dataset_FINAL.h5"

# Stats Config
INPUT_STATS_PATTERN = str(PROCESSED_DIR / "processing_stats_node_*.xlsx")
OUTPUT_MERGED_XLSX = PROCESSED_DIR / "FINAL_merged_processing_stats.xlsx"
OUTPUT_SUMMARY_TXT = PROCESSED_DIR / "FINAL_summary_metrics.txt"
OUTPUT_PIE_CHART = PROCESSED_DIR / "FINAL_global_ligand_class_pie.png"
OUTPUT_BAR_CHART = PROCESSED_DIR / "FINAL_global_ligand_bar.png"

# ----------------------------
# PART 0: LOAD CLASS LOOKUP
# ----------------------------
def load_class_lookup():
    """
    Loads the original metadata file to create a reliable lookup table.
    Returns: Dict { (pdb_id, ligand_name): 'ClassString' }
    """
    if not METADATA_FILE.exists():
        print(f"WARNING: Metadata file not found at {METADATA_FILE}")
        print("Classes will be guessed based on name only (less accurate).")
        return None

    print(f"Loading class definitions from {METADATA_FILE}...")
    try:
        df = pd.read_excel(METADATA_FILE)
        # Normalize keys for lookup
        df['PDB_ID'] = df['PDB_ID'].astype(str).str.lower().str.strip()
        
        lookup = {}
        for _, row in df.iterrows():
            pdb = row['PDB_ID']
            # The metadata file has "Ligand_Names" (e.g., "ATP, MG") 
            # and "Ligand_Classes" (e.g., "Cofactor, Ion")
            
            ligs = str(row.get('Ligand_Names', '')).split(',')
            classes = str(row.get('Ligand_Classes', '')).split(',')
            
            # Map them 1-to-1 if lengths match
            if len(ligs) == len(classes):
                for l, c in zip(ligs, classes):
                    lookup[(pdb, l.strip())] = c.strip()
            else:
                # Fallback: Assign the first class to all ligands if mismatch
                primary_class = classes[0].strip() if classes else "Other"
                for l in ligs:
                    lookup[(pdb, l.strip())] = primary_class
                    
        return lookup
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None

# ----------------------------
# PART 1: HDF5 MERGING
# ----------------------------
def merge_hdf5_datasets():
    print("\n" + "="*40)
    print(" PART 1: MERGING HDF5 DATASETS")
    print("="*40)

    files = sorted(glob.glob(str(INPUT_H5_PATTERN)))
    print(f"Found {len(files)} partition files to merge.")

    if not files:
        print("No HDF5 files found. Check your directory.")
        return

    # Create the final container
    with h5py.File(OUTPUT_FINAL_H5, 'w') as h5_out:
        
        # Initialize datasets using the structure of the first file
        print("Initializing output file structure...")
        with h5py.File(files[0], 'r') as f0:
            for key in f0.keys():
                shape = list(f0[key].shape)
                shape[0] = 0 
                max_shape = list(shape)
                max_shape[0] = None 
                
                h5_out.create_dataset(
                    key, 
                    shape=tuple(shape), 
                    maxshape=tuple(max_shape), 
                    dtype=f0[key].dtype,
                    chunks=True
                )

        # Append data from each part
        total_items = 0
        
        for fname in tqdm(files, desc="Merging H5 Files"):
            try:
                with h5py.File(fname, 'r') as h5_in:
                    current_size = h5_in['pdb_ids'].shape[0]
                    if current_size == 0: continue
                    
                    for key in h5_out.keys():
                        old_len = h5_out[key].shape[0]
                        new_len = old_len + current_size
                        h5_out[key].resize(new_len, axis=0)
                        h5_out[key][old_len:new_len] = h5_in[key][:]
                    
                    total_items += current_size
            except Exception as e:
                print(f"Error merging {fname}: {e}")

    print(f"\nSUCCESS! Merged {total_items} total items.")
    print(f"Final dataset saved to: {OUTPUT_FINAL_H5}")

# ----------------------------
# PART 2: STATS AGGREGATION
# ----------------------------
def parse_ligand_string(ligand_str):
    """Parses string like 'ATP, ATP, MG' into a list."""
    if pd.isna(ligand_str) or str(ligand_str).strip() == "":
        return []
    # Clean up list elements
    return [x.strip() for x in str(ligand_str).split(",")]

def aggregate_statistics():
    print("\n" + "="*40)
    print(" PART 2: AGGREGATING STATISTICS")
    print("="*40)

    # 1. LOAD CLASS LOOKUP (The "Smart" Data)
    class_lookup = load_class_lookup()

    # 2. FIND STATS FILES
    files = glob.glob(INPUT_STATS_PATTERN)
    if not files:
        print(f"No stats Excel files found matching {INPUT_STATS_PATTERN}")
        return

    print(f"Found {len(files)} stats files. Merging...")

    # 3. MERGE DATAFRAMES
    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_excel(f))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not df_list:
        print("No valid data found in Excel files.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # 4. CALCULATE METRICS
    total_pdbs_scanned = len(full_df)
    non_empty_df = full_df[full_df['Saved_Count'] > 0].copy()
    total_non_empty_pdbs = len(non_empty_df)
    total_ligands_saved = non_empty_df['Saved_Count'].sum()
    total_ligands_removed = full_df['Removed_Count'].sum()

    # 5. PARSE NAMES AND ASSIGN CLASSES
    all_saved_ligands = []
    all_saved_classes = []

    for _, row in non_empty_df.iterrows():
        # Get PDB ID to lookup metadata
        pdb_id = str(row['PDB_ID']).lower().strip()
        lig_str = row['Saved_Ligands']
        
        ligs = parse_ligand_string(lig_str)
        all_saved_ligands.extend(ligs)
        
        for lig in ligs:
            # 1. Try Metadata Lookup (Accurate RDKit result)
            if class_lookup and (pdb_id, lig) in class_lookup:
                found_class = class_lookup[(pdb_id, lig)]
            
            # 2. Fallback to Util (Name-based guessing)
            else:
                found_class = get_ligand_class_by_name(lig)
            
            all_saved_classes.append(found_class)
    
    ligand_counts = Counter(all_saved_ligands)
    class_counts = Counter(all_saved_classes)

    # 6. GENERATE TEXT REPORT
    report = (
        "========================================\n"
        "       GLOBAL DATASET STATISTICS        \n"
        "========================================\n"
        f"Total PDBs Scanned:       {total_pdbs_scanned}\n"
        f"Total Non-Empty PDBs:     {total_non_empty_pdbs}\n"
        f"Total Ligands Saved:      {total_ligands_saved}\n"
        f"Total Ligands Removed:    {total_ligands_removed}\n"
        "========================================\n"
        "LIGAND CLASS DISTRIBUTION:\n"
    )
    
    # Report Classes
    for cls, count in class_counts.most_common():
        pct = (count / total_ligands_saved) * 100 if total_ligands_saved > 0 else 0
        report += f"{cls}: {count} ({pct:.1f}%)\n"
    
    report += "\nTOP 30 SPECIFIC LIGANDS:\n"
    for lig, count in ligand_counts.most_common(30):
        report += f"{lig}: {count}\n"

    print(report)
    with open(OUTPUT_SUMMARY_TXT, "w") as f:
        f.write(report)
    print(f"Saved summary metrics to {OUTPUT_SUMMARY_TXT}")

    # 7. SAVE MERGED EXCEL
    full_df.to_excel(OUTPUT_MERGED_XLSX, index=False)
    print(f"Saved detailed merged stats to {OUTPUT_MERGED_XLSX}")

    # 8. GENERATE PIE CHART (CLASSES)
    # 
    if class_counts:
        sorted_classes = class_counts.most_common()
        labels = [x[0] for x in sorted_classes]
        values = [x[1] for x in sorted_classes]
        
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette('pastel')[0:len(labels)]
        explode = [0.05] + [0] * (len(labels) - 1)
        
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=140, explode=explode, shadow=True)
        plt.title(f"Global Ligand Class Distribution\n(Total: {int(total_ligands_saved)})")
        plt.tight_layout()
        plt.savefig(OUTPUT_PIE_CHART)
        print(f"Saved Class Pie Chart to {OUTPUT_PIE_CHART}")

    # 9. GENERATE BAR CHART (SPECIFIC LIGANDS)
    # 
    if ligand_counts:
        top_50 = ligand_counts.most_common(50)
        labels, values = zip(*top_50)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=list(labels), y=list(values), palette='viridis')
        plt.title(f"Top 50 Saved Ligands (Global Count)")
        plt.xticks(rotation=90)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUTPUT_BAR_CHART)
        print(f"Saved Bar Chart to {OUTPUT_BAR_CHART}")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    if not PROCESSED_DIR.exists():
        print(f"Error: Directory {PROCESSED_DIR} does not exist.")
        exit()

    merge_hdf5_datasets()
    aggregate_statistics()