import h5py
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
HDF5_PATH = "data/processed/ml_dataset_FINAL.h5"  # Check your path
OUTPUT_PATH = "data/processed/molformer_embeddings.pt"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def precompute_molformer():
    print(f"--- Loading MolFormer on {DEVICE} ---")
    # Using the 10% model as you requested (lighter/faster)
    # You can swap for 'ibm/MoLFormer-XL-both-10pct' if you want
    model_name = "ibm/MoLFormer-XL-both-10pct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    model.eval()

    print(f"--- Reading SMILES from {HDF5_PATH} ---")
    with h5py.File(HDF5_PATH, 'r') as f:
        # Decode bytes to strings
        all_smiles = [s.decode('utf-8') for s in f['ligand_smiles'][:]]
        
    print(f"--- Processing {len(all_smiles)} ligands... ---")
    
    all_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(all_smiles), BATCH_SIZE)):
        batch_smiles = all_smiles[i : i + BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(batch_smiles, padding=True, return_tensors="pt").to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            # Use pooler_output (CLS token) for the sequence representation
            # Shape: [batch_size, 768] usually
            embeddings = outputs.pooler_output.cpu()
            
        all_embeddings.append(embeddings)

    # Concatenate all batches
    final_tensor = torch.cat(all_embeddings, dim=0)
    print(f"--- Done! Shape: {final_tensor.shape} ---")
    
    # Save
    torch.save(final_tensor, OUTPUT_PATH)
    print(f"--- Saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    precompute_molformer()