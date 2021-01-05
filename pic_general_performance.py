import os
import h5py
import json

result_dir = '/home/tan/VideoIMUCombined/experiment_data/KAM/training_results/2021-01-04 21:39:33.127418'

with h5py.File(os.path.join(result_dir, 'results.h5'), 'r') as hf:
    _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
    _data_fields = json.loads(hf.attrs['columns'])

print(1)
