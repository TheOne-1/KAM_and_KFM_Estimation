import h5py
import os
import json
from const import SUBJECTS, DATA_PATH, EXAMPLE_DATA_FIELDS, VIDEO_LIST
import torch
import shutil
import pandas as pd
import numpy as np


def vid_static_cali():
    vid_y_90_col_loc = [data_fields.index(marker + '_y_90') for marker in VIDEO_LIST]
    for sub_name, sub_data in data_all_sub.items():
        static_side_df = pd.read_csv(DATA_PATH + '/' + sub_name + '/combined/static_side.csv', index_col=0)
        r_ankle_z = np.mean(static_side_df['RAnkle_y_90'])
        sub_data[:, :, vid_y_90_col_loc] = sub_data[:, :, vid_y_90_col_loc] - r_ankle_z + 1500
        data_all_sub[sub_name] = sub_data


"""Create data file"""
with h5py.File(os.environ.get('KAM_DATA_PATH') + '/40samples+stance.h5', 'r') as hf:
    data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
    data_fields = json.loads(hf.attrs['columns'])
    export_path = 'trained_models_and_example_data/example_data.h5'
    example_col_loc = [data_fields.index(column) for column in EXAMPLE_DATA_FIELDS]
    vid_static_cali()
    with h5py.File(export_path, 'w') as hf:
        hf.create_dataset('subject_01', data=data_all_sub[SUBJECTS[5]][200:210, :, example_col_loc], dtype='float32')
        hf.create_dataset('subject_02', data=data_all_sub[SUBJECTS[6]][200:210, :, example_col_loc], dtype='float32')
        hf.attrs['columns'] = json.dumps(EXAMPLE_DATA_FIELDS)


"""Create model file"""
test_name = '0307'
subject_folder = 's020_houjikang'
for moment in ['KAM', 'KFM']:
    for model, model_name in zip(['8IMU_2camera', '8IMU', '2camera'], ['fusion', 'IMU_based', 'camera_based']):
        model_path_src = os.path.join(DATA_PATH, 'training_results', test_name+moment, model, 'sub_models',
                                      subject_folder, 'four_source_model.pth')
        model_path_dest = './trained_models_and_example_data/' + model_name + '_' + moment + '.pth'
        shutil.copyfile(model_path_src, model_path_dest)
