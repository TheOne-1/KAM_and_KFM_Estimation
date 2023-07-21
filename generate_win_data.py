import os
import pandas as pd
import numpy as np
import h5py
import json
from customized_logger import logger as logging
from const import TRIALS, SUBJECTS, KAM_PHASE, FORCE_PHASE, STEP_PHASE
from const import R_PLATE_FORCE_Z, R_KAM_COLUMN, EVENT_COLUMN, CONTINUOUS_FIELDS, DISCRETE_FIELDS, SEGMENT_DEFINITIONS
from const import SEGMENT_DATA_FIELDS, STEP_TYPE, STANCE_SWING, STANCE
from wearable_toolkit import data_filter
import matplotlib.pyplot as plt


anthro_fields = ['body weight', 'body height', 'subject_id', 'trial_id']
kinetic_fields = [
    'EXT_KM_X', 'EXT_KM_Y', 'EXT_KM_Z',
    'plate_1_force_x', 'plate_1_force_y', 'plate_1_force_z', 'plate_1_cop_x', 'plate_1_cop_y', 'plate_1_cop_z',
    'plate_2_force_x', 'plate_2_force_y', 'plate_2_force_z', 'plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z']
imu_fields = [
    'AccelX_L_FOOT', 'AccelY_L_FOOT', 'AccelZ_L_FOOT', 'GyroX_L_FOOT', 'GyroY_L_FOOT', 'GyroZ_L_FOOT', 'MagX_L_FOOT', 'MagY_L_FOOT', 'MagZ_L_FOOT',
    'AccelX_R_FOOT', 'AccelY_R_FOOT', 'AccelZ_R_FOOT', 'GyroX_R_FOOT', 'GyroY_R_FOOT', 'GyroZ_R_FOOT', 'MagX_R_FOOT', 'MagY_R_FOOT', 'MagZ_R_FOOT',
    'AccelX_R_SHANK', 'AccelY_R_SHANK', 'AccelZ_R_SHANK', 'GyroX_R_SHANK', 'GyroY_R_SHANK', 'GyroZ_R_SHANK', 'MagX_R_SHANK', 'MagY_R_SHANK', 'MagZ_R_SHANK',
    'AccelX_R_THIGH', 'AccelY_R_THIGH', 'AccelZ_R_THIGH', 'GyroX_R_THIGH', 'GyroY_R_THIGH', 'GyroZ_R_THIGH', 'MagX_R_THIGH', 'MagY_R_THIGH', 'MagZ_R_THIGH',
    'AccelX_WAIST', 'AccelY_WAIST', 'AccelZ_WAIST', 'GyroX_WAIST', 'GyroY_WAIST', 'GyroZ_WAIST', 'MagX_WAIST', 'MagY_WAIST', 'MagZ_WAIST',
    'AccelX_CHEST', 'AccelY_CHEST', 'AccelZ_CHEST', 'GyroX_CHEST', 'GyroY_CHEST', 'GyroZ_CHEST', 'MagX_CHEST', 'MagY_CHEST', 'MagZ_CHEST',
    'AccelX_L_SHANK', 'AccelY_L_SHANK', 'AccelZ_L_SHANK', 'GyroX_L_SHANK', 'GyroY_L_SHANK', 'GyroZ_L_SHANK', 'MagX_L_SHANK', 'MagY_L_SHANK', 'MagZ_L_SHANK',
    'AccelX_L_THIGH', 'AccelY_L_THIGH', 'AccelZ_L_THIGH', 'GyroX_L_THIGH', 'GyroY_L_THIGH', 'GyroZ_L_THIGH', 'MagX_L_THIGH', 'MagY_L_THIGH', 'MagZ_L_THIGH']
fields_to_keep = anthro_fields + kinetic_fields + imu_fields


def rename_kam_dataset(export_path):
    with h5py.File(export_path, 'r+') as hf:
        data_columns = json.loads(hf.attrs['columns'])
        for i, column in enumerate(data_columns):
            if 'Accel' == column[:5]:
                axis, segment = column.split('Accel')[1][0], column.split('Accel')[1][2:]
                column_new = segment + '_Accel_' + axis
                data_columns[i] = column_new
            if 'Gyro' == column[:4]:
                axis, segment = column.split('Gyro')[1][0], column.split('Gyro')[1][2:]
                column_new = segment + '_Gyro_' + axis
                data_columns[i] = column_new
            if 'EXT_KM_X' == column:
                column_new = 'KFM'
                data_columns[i] = column_new
            if 'EXT_KM_Y' == column:
                column_new = 'KAM'
                data_columns[i] = column_new
        if 'sub_id' not in data_columns:
            for sub_name, data in hf.items():
                data_ = data[:]
                subject_id_array = np.full([data_.shape[0], data_.shape[1], 1], int(sub_name[-2:])-1)
                data_ = np.concatenate([subject_id_array, data_], axis=2)
                del hf[sub_name]
                hf.create_dataset(sub_name, data_.shape, data=data_)
            data_columns.insert(0, 'sub_id')
        hf.attrs['columns'] = json.dumps(data_columns)


def get_step_data(data_df):
    def get_segment_point(seg_axis):
        seg_name, axis = seg_axis[:-2], seg_axis[-1:]
        markers = SEGMENT_DEFINITIONS[seg_name]
        markers_in_one_direction = list(map(lambda x: x + '_' + axis, markers))
        return pd.DataFrame(data_df[markers_in_one_direction].mean(axis=1), columns=[seg_axis])

    data_df = pd.concat([data_df] + list(map(get_segment_point, SEGMENT_DATA_FIELDS)), axis=1)

    imu_data = data_df[imu_fields].values
    imu_data_filtered = data_filter(imu_data, 15, 100, 4)
    data_df[imu_fields] = imu_data_filtered

    win_list = []
    win_len, win_step = 128, 64        # 1.28 second windows with 50% overlapping
    walking_index = np.where(~np.isnan(data_df[EVENT_COLUMN]))[0]
    win_start, win_end = walking_index[0], walking_index[-1]
    for index_ in range(win_start, win_end - win_len, win_step):
        step_df = data_df.iloc[index_:index_+win_len]
        step_df.index = range(step_df.shape[0])
        win_list.append(step_df)
    return win_list


def generate_step_data(export_path, processes):
    # create training data and test data
    subject_trial_dict = {}
    for subject_trial, data_df in all_data_dict.items():
        data_df = data_df.rename(columns={'Unnamed: 0': 'sample_index'})
        subject_trial_dict[subject_trial] = get_step_data(data_df)
        for f in processes:
            subject_trial_dict[subject_trial] = f(subject_trial_dict[subject_trial])

    subject_data_dict = {subject: [data for trial in TRIALS for data in subject_trial_dict[subject + " " + trial]]
                         for subject in SUBJECTS}
    with h5py.File(export_path, 'w') as hf:
        for i_subject, (subject, data_collections) in enumerate(subject_data_dict.items()):
            subject_whole_trial = np.concatenate(
                [np.array(data.loc[:, fields_to_keep])[np.newaxis, :, :] for data in data_collections], axis=0)

            if i_subject < 9:
                sub_num_str = 'subject_0' + str(i_subject+1)
            else:
                sub_num_str = 'subject_' + str(i_subject+1)
            hf.create_dataset(sub_num_str, data=subject_whole_trial, dtype='float32')
        hf.attrs['columns'] = json.dumps(fields_to_keep)


if __name__ == "__main__":
    logging.debug("Loading all csv data")
    DATA_PATH = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2021_VideoIMUCombined/experiment_data/KAM/'
    export_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/filtered_walking_knee_moment.h5'
    all_data_dict = {subject + " " + trial: pd.read_csv(os.path.join(DATA_PATH, subject, "combined", trial + ".csv"))
                     for subject in SUBJECTS for trial in TRIALS}
    custom_process = []
    generate_step_data(export_path, custom_process)
    rename_kam_dataset(export_path)
