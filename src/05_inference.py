import argparse
import torch
import mrcfile
import gemmi
import numpy as np
from architecture import SCUNet, smiles_to_embedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess(map_path, target_voxel=0.5):
    m = gemmi.read_ccp4_map(map_path)
    m.setup(0.0, gemmi.MapSetup.Full)
    
    # Resample
    new_grid = gemmi.FloatGrid(
        int(m.grid.unit_cell.a/target_voxel), 
        int(m.grid.unit_cell.b/target_voxel), 
        int(m.grid.unit_cell.c/target_voxel)
    )
    new_grid.set_unit_cell(gemmi.UnitCell(new_grid.nu*target_voxel, new_grid.nv*target_voxel, new_grid.nw*target_voxel, 90,90,90))
    gemmi.interpolate_grid(new_grid, m.grid, gemmi.Transform(), order=2)
    
    # Normalize & Center Crop
    arr = np.array(new_grid, copy=False)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    
    cx, cy, cz = np.array(arr.shape)//2
    crop = arr[cx-48:cx+48, cy-48:cy+48, cz-48:cz+48]
    
    return torch.from_numpy(crop).float().unsqueeze(0).unsqueeze(0), new_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--out", default="pred.mrc")
    args = parser.parse_args()

    model = SCUNet(in_nc=2).to(DEVICE)
    model.load_state_dict(torch.load("../checkpoints/best_model.pth", map_location=DEVICE))
    model.eval()

    vol, ref_grid = preprocess(args.map)
    emb = torch.from_numpy(smiles_to_embedding(args.smiles)).float().unsqueeze(0)
    
    # Create Dummy Protein Mask (Channel 0) + Experimental Density (Channel 1)
    # Since we don't have a protein mask at inference, we pass zeros or the map itself
    # Better approach: Pass exp density in both channels if mask missing
    inputs = torch.cat([vol, vol], dim=1).to(DEVICE)
    
    with torch.no_grad():
        out = model(inputs, emb.to(DEVICE))
    
    with mrcfile.new(args.out, overwrite=True) as mrc:
        mrc.set_data(out.squeeze().cpu().numpy())
        mrc.voxel_size = 0.5

if __name__ == "__main__":
    main()
