# CryoLigate
Understanding protein-ligand interactions at the atomic level is crucial for unraveling how
ligands influence macromolecular functions. This insight can be applied to the development
of pharmaceuticals through structure-based drug discovery. However, traditional X-ray crys-
tallographic methods for obtaining experimental structures of such complexes are challenging
and time-intensive. Recent advances in single-particle cryo-EM have addressed this challenge,
enabling the determination of atomic-resolution structures of complex biomolecular systems.
While cryo-EM now provides exceptional resolution for overall structures (often below 2 ˚A),
ligand resolutions remain too low for accurate modeling. Recently, artificial intelligent (AI)-
based methods have been developed to model and refine structures based on EM data, however,
their primary focus has been on protein accuracy. Here, we exploited the increased power of
data-driven research and build an AI model to improve low resolution ligand maps and refine
structural models for small molecules in protein-ligand complexes

1. **Install:** `conda create -n CryoLigate python=3.9`
`conda activate CryoLigate`
`pip install -r requirements.txt`
2. **Fetch:** `python src/01_fetch_metadata.py`
3. **Download:** `python src/02_download_raw_data.py`
4. **Build:** `python src/03_build_processed_dataset.py`
5. **Train:** `python src/04_train.py`
6. **Infer:** `python src/05_inference.py --map my_map.map --smiles "CCO" --out result.mrc`
