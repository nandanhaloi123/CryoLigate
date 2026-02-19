import mrcfile
import gemmi
import numpy as np

def resample_em_map(gemmi_grid, target_voxel):
    """
    Resamples a gemmi grid to a specific voxel size WITHOUT changing physical dimensions.
    """
    u = gemmi_grid.unit_cell
    
    # Calculate new grid dimensions
    new_nu = int(round(u.a / target_voxel))
    new_nv = int(round(u.b / target_voxel))
    new_nw = int(round(u.c / target_voxel))
    
    # Create new grid with ORIGINAL Unit Cell dimensions
    new_grid = gemmi.FloatGrid(new_nu, new_nv, new_nw)
    new_grid.set_unit_cell(u)
    
    # Interpolate
    gemmi.interpolate_grid(new_grid, gemmi_grid, gemmi.Transform(), order=3)
    return new_grid

def coord_to_grid_index(coord, grid):
    """Converts Angstrom coordinates to Grid Indices."""
    pos = gemmi.Position(coord[0], coord[1], coord[2])
    fractional = grid.unit_cell.fractionalize(pos)
    return np.array([
        int(round(fractional.x * grid.nu)),
        int(round(fractional.y * grid.nv)),
        int(round(fractional.z * grid.nw)),
    ], dtype=np.int32)

def save_mrc_with_origin(data, filepath, unit_cell, origin_angstroms):
    """
    Saves a numpy array as .mrc with correct PHYSICAL origin and DIMENSIONS.
    
    Args:
        data: (Z, Y, X) numpy array (Note: Pre-transpose if you have X,Y,Z)
        filepath: Path to save
        unit_cell: gemmi.UnitCell object (Crucial for correct alignment)
        origin_angstroms: (x,y,z) float tuple
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with mrcfile.new(str(filepath), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        
        # --- CRITICAL FIX: Write exact Unit Cell dimensions ---
        # This allows viewers to calculate the exact voxel size for each axis (X,Y,Z)
        # preventing the "drift" or "shift" caused by anisotropic voxels.
        mrc.header.cella.x = unit_cell.a
        mrc.header.cella.y = unit_cell.b
        mrc.header.cella.z = unit_cell.c
        
        # Calculate isotropic voxel size estimate for the header stats (optional but good)
        # But the 'cella' fields above are what ChimeraX uses for alignment.
        mrc.voxel_size = unit_cell.a / data.shape[2] # Shape is Z,Y,X

        # Set the Origin
        mrc.header.origin.x = float(origin_angstroms[0])
        mrc.header.origin.y = float(origin_angstroms[1])
        mrc.header.origin.z = float(origin_angstroms[2])
        
        mrc.update_header_stats()
        
        



from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ----------------------------
# CHEMICAL LISTS
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

SUGARS = {
    'NAG', 'NDG', 'MAN', 'BMA', 'GAL', 'GLA', 'FUC', 'SIA', 'FUL', 'XYP', 'GLC', 'SUC', 'TRE', 'NGA', 'AHR'
}

COFACTORS = {
    'HEM', 'HEA', 'FAD', 'FMN', 'NAD', 'NAP', 'ADP', 'ATP', 'GTP', 'GDP', 'AMP', 'SAM', 'SAH', 'TPP', 'PLP'
}

AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEP', 'TPO', 'PTR', 'PCA'
}

BUFFERS_SOLVENTS = {
    'HOH', 'DOD', 'EDO', 'GOL', 'PEG', 'PG4', 'PGE', 'MES', 'HEPES', 'TRIS', 'EPE', 'ACY', 'FMT', 'ACT'
}

EXCLUDE_LIST = IONS | BUFFERS_SOLVENTS | SUGARS | LIPIDS_DETERGENTS

# ----------------------------
# CLASSIFICATION FUNCTIONS
# ----------------------------

def get_ligand_class_by_name(ccd_code):
    """
    Fast lookup based ONLY on the 3-letter CCD code.
    Used when you don't have SMILES (e.g., in final stats aggregation).
    """
    code = str(ccd_code).upper().strip()
    
    # Check explicit lists
    if code in AMINO_ACIDS: return "Amino Acids"
    if code in SUGARS: return "Saccharides"
    if code in COFACTORS: return "Cofactors/Nucleotides"
    if code in LIPIDS_DETERGENTS: return "Lipids/Detergents"
    if code in IONS: return "Ions (Excluded)"
    if code in BUFFERS_SOLVENTS: return "Buffers/Solvents (Excluded)"
    
    return "Drug-like/Other"


def classify_ligand_smart(ccd_code, smiles=None, qed=0.0):
    """
    Robust classification using RDKit if available.
    Used during metadata fetching/dataset building.
    """
    code = str(ccd_code).upper().strip()
    
    # 1. Check explicit lists first (fastest)
    basic_class = get_ligand_class_by_name(code)
    if basic_class != "Drug-like/Other":
        return basic_class

    # 2. Use RDKit heuristics for edge cases (e.g. unknown lipids)
    if RDKIT_AVAILABLE and smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                
                # Heuristic: Long flexible chains are usually lipids/detergents
                if rot_bonds > 12 and mw > 250: 
                    return "Lipids/Detergents"
                
                # Heuristic: Tiny organics
                if mw < 300: 
                    return "Fragments/Metabolites"
        except: 
            pass 

    # 3. Fallback to QED or Default
    if qed >= 0.4: 
        return "Drug-like"
    
    return "Other"