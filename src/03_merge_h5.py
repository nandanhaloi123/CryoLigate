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
# CONFIGURATION
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# HDF5 Config
INPUT_H5_PATTERN = PROCESSED_DIR / "ml_dataset_part_*.h5"
OUTPUT_FINAL_H5 = PROCESSED_DIR / "ml_dataset_FINAL.h5"

# Stats Config
INPUT_STATS_PATTERN = str(PROCESSED_DIR / "processing_stats_node_*.xlsx")
OUTPUT_MERGED_XLSX = PROCESSED_DIR / "FINAL_merged_processing_stats.xlsx"
OUTPUT_SUMMARY_TXT = PROCESSED_DIR / "FINAL_summary_metrics.txt"
OUTPUT_PIE_CHART = PROCESSED_DIR / "FINAL_global_ligand_pie.png"
OUTPUT_BAR_CHART = PROCESSED_DIR / "FINAL_global_ligand_bar.png"

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
                # Get shape (e.g., 100, 2, 64, 64, 64)
                shape = list(f0[key].shape)
                
                # Set first dimension to 0 (we will grow this dynamically)
                shape[0] = 0 
                
                # Set maxshape to None for the first dim (allows resizing)
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
                    # Check how many valid items are in this specific part
                    current_size = h5_in['pdb_ids'].shape[0]
                    if current_size == 0: continue
                    
                    for key in h5_out.keys():
                        # 1. Resize output to accommodate new data
                        old_len = h5_out[key].shape[0]
                        new_len = old_len + current_size
                        h5_out[key].resize(new_len, axis=0)
                        
                        # 2. Write data into the new empty space
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
    return [x.strip() for x in str(ligand_str).split(",")]

def aggregate_statistics():
    print("\n" + "="*40)
    print(" PART 2: AGGREGATING STATISTICS")
    print("="*40)

    # 1. FIND FILES
    files = glob.glob(INPUT_STATS_PATTERN)
    if not files:
        print(f"No stats Excel files found matching {INPUT_STATS_PATTERN}")
        return

    print(f"Found {len(files)} stats files. Merging...")

    # 2. LOAD AND MERGE
    df_list = []
    for f in files:
        try:
            df = pd.read_excel(f)
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not df_list:
        print("No valid data found in Excel files.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # 3. CALCULATE METRICS
    total_pdbs_scanned = len(full_df)
    
    # "Non-Empty PDBs" are those where at least 1 ligand was SAVED
    non_empty_df = full_df[full_df['Saved_Count'] > 0].copy()
    total_non_empty_pdbs = len(non_empty_df)
    
    total_ligands_saved = non_empty_df['Saved_Count'].sum()
    total_ligands_removed = full_df['Removed_Count'].sum()

    # 4. PARSE LIGAND NAMES FOR DISTRIBUTION
    all_saved_ligands = []
    for lig_str in non_empty_df['Saved_Ligands']:
        all_saved_ligands.extend(parse_ligand_string(lig_str))
    
    ligand_counts = Counter(all_saved_ligands)

    # 5. GENERATE TEXT REPORT
    report = (
        "========================================\n"
        "       GLOBAL DATASET STATISTICS        \n"
        "========================================\n"
        f"Total PDBs Scanned:       {total_pdbs_scanned}\n"
        f"Total Non-Empty PDBs:     {total_non_empty_pdbs} (Files with at least 1 valid ligand)\n"
        f"Total Ligands Saved:      {total_ligands_saved}\n"
        f"Total Ligands Removed:    {total_ligands_removed}\n"
        "========================================\n"
        "TOP 30 SAVED LIGANDS:\n"
    )
    
    for lig, count in ligand_counts.most_common(30):
        report += f"{lig}: {count}\n"

    print(report)
    
    with open(OUTPUT_SUMMARY_TXT, "w") as f:
        f.write(report)
    print(f"Saved summary metrics to {OUTPUT_SUMMARY_TXT}")

    # 6. SAVE MERGED EXCEL
    # This Excel contains every PDB, what was kept, and what was thrown away.
    full_df.to_excel(OUTPUT_MERGED_XLSX, index=False)
    print(f"Saved detailed merged stats to {OUTPUT_MERGED_XLSX}")

    # 7. GENERATE PIE CHART
    if ligand_counts:
        top_n = 14
        most_common = ligand_counts.most_common(top_n)
        top_labels = [x[0] for x in most_common]
        top_values = [x[1] for x in most_common]
        
        total_top = sum(top_values)
        total_all = sum(ligand_counts.values())
        others_count = total_all - total_top
        
        if others_count > 0:
            top_labels.append("Others")
            top_values.append(others_count)

        plt.figure(figsize=(10, 8))
        # Use a nice color palette
        colors = sns.color_palette('pastel')[0:len(top_labels)]
        
        plt.pie(top_values, labels=top_labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title(f"Global Ligand Distribution (Total: {int(total_ligands_saved)})")
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(OUTPUT_PIE_CHART)
        print(f"Saved Pie Chart to {OUTPUT_PIE_CHART}")

    # 8. GENERATE BAR CHART
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
    # Ensure processed directory exists (it should if you ran the previous script)
    if not PROCESSED_DIR.exists():
        print(f"Error: Directory {PROCESSED_DIR} does not exist.")
        exit()

    # Run the merger
    merge_hdf5_datasets()
    
    # Run the analytics
    aggregate_statistics()