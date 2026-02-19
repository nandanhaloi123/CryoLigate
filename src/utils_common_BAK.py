import mrcfile
import gemmi
import numpy as np
import scipy.ndimage

def resample_em_map(gemmi_grid, target_voxel):
    """
    Resamples a gemmi grid to a specific voxel size.
    Returns: gemmi.FloatGrid
    """
    new_size = [
        int(round(gemmi_grid.unit_cell.a / target_voxel)),
        int(round(gemmi_grid.unit_cell.b / target_voxel)),
        int(round(gemmi_grid.unit_cell.c / target_voxel))
    ]
    new_grid = gemmi.FloatGrid(*new_size)
    new_grid.set_unit_cell(gemmi.UnitCell(
        new_size[0] * target_voxel,
        new_size[1] * target_voxel,
        new_size[2] * target_voxel,
        90, 90, 90
    ))
    gemmi.interpolate_grid(new_grid, gemmi_grid, gemmi.Transform(), order=3)
    return new_grid

def coord_to_grid_index(coord, grid):
    """
    Converts Angstrom coordinates (x,y,z) to Grid Indices (i,j,k).
    """
    pos = gemmi.Position(coord[0], coord[1], coord[2])
    fractional = grid.unit_cell.fractionalize(pos)
    return np.array([
        int(round(fractional.x * grid.nu)),
        int(round(fractional.y * grid.nv)),
        int(round(fractional.z * grid.nw)),
    ], dtype=np.int32)

def save_mrc_with_origin(data, filepath, voxel_size, origin_angstroms):
    """
    Saves a numpy array as .mrc with the correct PHYSICAL origin header.
    This ensures it aligns with the original PDB/Map in ChimeraX.
    
    data: (D, H, W) numpy array
    filepath: path to save
    voxel_size: float (e.g. 0.5)
    origin_angstroms: tuple/list (x, y, z) in Angstroms
    """
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with mrcfile.new(str(filepath), overwrite=True) as mrc:
        # MRC standard expects (Z, Y, X) order usually, but Gemmi/Numpy often work in (X,Y,Z).
        # We assume data is (X,Y,Z). mrcfile.set_data expects C-order (Z,Y,X) or F-order.
        # Transposing usually fixes the rotation issue.
        mrc.set_data(data.astype(np.float32))
        
        mrc.voxel_size = voxel_size
        
        # KEY STEP: Set the Origin
        mrc.header.origin.x = float(origin_angstroms[0])
        mrc.header.origin.y = float(origin_angstroms[1])
        mrc.header.origin.z = float(origin_angstroms[2])
        
        mrc.update_header_stats()