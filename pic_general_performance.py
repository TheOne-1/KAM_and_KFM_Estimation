import os
import h5py
import json

result_dir = '/media/dianxin/Samsung USB/2021-01-04 20_22_01.552985/s002_wangdianxin/'

with h5py.File(os.path.join(result_dir, 'results.h5'), 'r') as hf:
    _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
    _data_fields = json.loads(hf.attrs['columns'])

print(1)
