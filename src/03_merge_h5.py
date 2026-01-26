import h5py
import glob
import os
import numpy as np
from tqdm import tqdm

# CONFIGURATION
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = "data/processed/ml_dataset_FINAL.h5"

def merge_datasets():
    # Find all partial files (e.g., ml_dataset_part_0.h5, part_1.h5...)
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "ml_dataset_part_*.h5")))
    print(f"Found {len(files)} partition files to merge.")

    if not files:
        print("No files found. Check your directory.")
        exit()

    # Create the final container
    with h5py.File(OUTPUT_FILE, 'w') as h5_out:
        
        # Initialize datasets using the structure of the first file
        print("Initializing output file...")
        with h5py.File(files[0], 'r') as f0:
            for key in f0.keys():
                # get shape (e.g., 100, 2, 96, 96, 96)
                shape = list(f0[key].shape)
                
                # Set first dimension to 0 (we will grow this)
                shape[0] = 0 
                
                # Set maxshape to None for the first dim (allows infinite resizing)
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
        
        for fname in tqdm(files, desc="Merging Files"):
            try:
                with h5py.File(fname, 'r') as h5_in:
                    # How many items in this specific file?
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
    print(f"Final dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_datasets()