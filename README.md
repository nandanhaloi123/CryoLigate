# CryoLigate

1. **Install:** `conda create -n CryoLigate python=3.9`
`conda activate CryoLigate`
`pip install -r requirements.txt`
2. **Fetch:** `python src/01_fetch_metadata.py`
3. **Download:** `python src/02_download_raw_data.py`
4. **Build:** `python src/03_build_processed_dataset.py`
5. **Train:** `python src/04_train.py`
6. **Infer:** `python src/05_inference.py --map my_map.map --smiles "CCO" --out result.mrc`
