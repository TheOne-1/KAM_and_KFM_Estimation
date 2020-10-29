import os
import pandas as pd
import numpy as np
import h5py
import json
from const import TRIALS, SUBJECTS, ALL_FIELDS
from const import SAMPLES_BEFORE_STEP
from config import DATA_PATH


def filter_and_clip_data(middle_data, max_len):
    # count the maximum time steps for each gati step
    def form_array_list(array, column):
        for _id in array[column].drop_duplicates().dropna():
            # fetch data before a step
            c = array[array[column] == _id].index
            b_min = c.min() - SAMPLES_BEFORE_STEP
            b_max = c.max()
            if b_min < array.index.min():
                continue
            b = array.loc[b_min:b_max, :]
            if b.shape[0] > max_len + SAMPLES_BEFORE_STEP:
                print("dropping step {} with size {} as it exceeds the limit {}".format(_id, b.shape[0], max_len))
                continue
            if (b[column] < 0).any():
                continue
            b.index = range(b.shape[0])
            b = b.reindex(range(max_len + SAMPLES_BEFORE_STEP))
            yield b

    step_list = list(form_array_list(middle_data, 'Event'))
    return step_list


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
    with h5py.File(export_path, 'w') as hf:
        for subject, data_collections in subject_data_dict.items():
            subject_whole_trial = np.concatenate(
                [np.array(data.loc[:, ALL_FIELDS])[np.newaxis, :, :] for data in
                 data_collections], axis=0)
            hf.create_dataset(subject, data=subject_whole_trial, dtype='float32')
        hf.attrs['columns'] = json.dumps(ALL_FIELDS)


def get_step_data(import_path):
    with h5py.File(import_path, 'r') as hf:
        subject_data = {subject: hf[subject][:] for subject in SUBJECTS}
    return subject_data


if __name__ == "__main__":
    generate_step_data(DATA_PATH + '40samples+stance_swing.h5')
