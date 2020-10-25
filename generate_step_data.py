import os
import pandas as pd
import numpy as np
import h5py
from const import TRIALS, SUBJECTS, IMU_DATA_FIELDS, IMU_FIELDS, VIDEO_LIST, VIDEO_DATA_FIELDS, TARGETS_LIST, MAX_LENGTH
from config import DATA_PATH


def filter_and_clip_data(middle_data, max_len):
    # remove data before the first event and the last event
    begin_index = middle_data[~middle_data['Event'].isnull()].index.min()
    end_index = middle_data[~middle_data['Event'].isnull()].index.max()
    middle_data = middle_data.loc[begin_index:(end_index + 1)]

    # drop true event as it it not used
    middle_data = middle_data.drop(columns=['True_Event'])

    # drop those events that is minus
    steps = middle_data['Event'].dropna().drop_duplicates()
    drop_steps = filter(lambda x: x < 0, steps)
    print("dropping steps {} as it is minus".format(list(drop_steps)))
    for step in drop_steps:
        middle_data[middle_data['Event'] == step] = np.nan
    middle_data = middle_data.dropna(subset=['Event'])

    # count the maximum time steps for each gati step
    def form_array_list(array, column):
        for _id in array[column].drop_duplicates():
            b = array[array[column] == _id]
            if b.shape[0] > max_len:
                print("dropping step {} with size {} as it exceeds the limit {}".format(_id, b.shape[0], max_len))
                continue
            b.index = range(b.shape[0])
            b = b.reindex(range(max_len))
            yield b

    a = list(form_array_list(middle_data, 'Event'))
    return a


def generate_subject_data():
    # create training data and test data
    all_data_dict = {subject + " " + trial: pd.read_csv(os.path.join(DATA_PATH, subject, "combined", trial + ".csv"))
                     for subject in SUBJECTS for trial in TRIALS}
    max_step_length = max([trial_data['Event'].value_counts().max() for trial_data in all_data_dict.values()])
    all_data_dict = {subject_trial: filter_and_clip_data(data, max_step_length)
                     for subject_trial, data in all_data_dict.items()}
    subject_dict = {subject: [data for trial in TRIALS for data in all_data_dict[subject + " " + trial]]
                    for subject in SUBJECTS}
    return subject_dict


def generate_step_data(export_path):
    subject_data_dict = generate_subject_data()

    # TODO: move this to preprocess data section
    # normalize video data
    for subject, data_collections in subject_data_dict.items():
        for data in data_collections:
            for angle in ["90", "180"]:
                for position in ["x", "y"]:
                    angle_specific_video_data_fields = [VIDEO + "_" + position + "_" + angle for VIDEO in VIDEO_LIST]
                    data.loc[:, angle_specific_video_data_fields] -= \
                        data.loc[:, "MidHip_" + position + "_" + angle].mean(axis=0)
                    data.loc[:, angle_specific_video_data_fields] /= 1920
                    data.loc[:, angle_specific_video_data_fields] += 0.5

    '''
    for subject, data_collections in subject_data_dict.items():
        for data in data_collections:
            data.loc[:, IMU_DATA_FIELDS] -= data.loc[:, IMU_DATA_FIELDS].mean(axis=0)
            data.loc[:, IMU_DATA_FIELDS] /= data.loc[:, IMU_DATA_FIELDS].std(axis=0)
    '''
    with h5py.File(export_path, 'w') as hf:
        for subject, data_collections in subject_data_dict.items():
            subject_whole_trial = np.concatenate(
                [np.array(data.loc[:, IMU_DATA_FIELDS + VIDEO_DATA_FIELDS + TARGETS_LIST])[np.newaxis, :, :] for data in
                 data_collections], axis=0)
            hf.create_dataset(subject, data=subject_whole_trial, dtype='float32')


def get_step_data(import_path):
    with h5py.File(import_path, 'r') as hf:
        subject_data = {subject: hf[subject][:] for subject in SUBJECTS}
    return subject_data


if __name__ == "__main__":
    generate_step_data(DATA_PATH + 'whole_data_160.h5')
