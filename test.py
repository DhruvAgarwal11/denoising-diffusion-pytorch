import h5py
with h5py.File("/home/dagarwal/processed_articulatory_features.h5", "r") as hf:
    modified_data = hf[f"chunks_{18000}"][0] 
    print(modified_data)