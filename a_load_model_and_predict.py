import h5py
import os
import json


with h5py.File('trained_models_and_example_data/example_data.h5', 'r') as hf:
    data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
    data_fields = json.loads(hf.attrs['columns'])
    x = 1