import os
import pandas as pd
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from customized_logger import logger as logging
from const import TRIALS, SUBJECTS, ALL_FIELDS, KAM_PHASE, FORCE_PHASE, STEP_PHASE
from const import SAMPLES_BEFORE_STEP, DATA_PATH, L_PLATE_FORCE_Z
from const import PADDING_MODE, PADDING_ZERO, PADDING_NEXT_STEP
from const import R_PLATE_FORCE_Z, R_KAM_COLUMN, EVENT_COLUMN, CONTINUOUS_FIELDS, DISCRETE_FIELDS


def get_step_data(step_array):
    def get_stance_swing_data(_id):
        stance_swing_step = step_array[np.abs(step_array[EVENT_COLUMN]) == _id].copy()
        stance_swing_step[STEP_PHASE] = 1.
        min_index = stance_swing_step.index.min()
        max_index = stance_swing_step.index.max()
        pre_step = step_array[min_index - SAMPLES_BEFORE_STEP: min_index].copy()
        pre_step[STEP_PHASE] = 0.
        if PADDING_MODE == PADDING_NEXT_STEP:
            post_step = step_array[max_index: max_step_length + min_index]
        elif PADDING_MODE == PADDING_ZERO:
            post_step_shape = [max_step_length + min_index - max_index, stance_swing_step.shape[1]]
            post_step = pd.DataFrame(np.zeros(post_step_shape), columns=stance_swing_step.columns)
        else:
            raise RuntimeError("None implement")

        step_data = pd.concat([pre_step, stance_swing_step, post_step])
        step_data[np.isnan(step_data)] = 0.
        step_data.index = range(step_data.shape[0])
        return step_data

    target_ids = np.abs(step_array[EVENT_COLUMN]).drop_duplicates().dropna()
    return map(get_stance_swing_data, target_ids)


def append_force_phase(one_step_array):
    # -20: swing phase might contain stance phase of next step, in which case, there might be a force.
    step_max_index = one_step_array[one_step_array[STEP_PHASE] == 1.].index.max() - 20
    padding_size = one_step_array.shape[0] - step_max_index
    one_step_array[FORCE_PHASE] = \
        np.pad(np.where(one_step_array[R_PLATE_FORCE_Z][:step_max_index] < -20, 1., 0.), (0, padding_size)) \
        * one_step_array[STEP_PHASE]
    return one_step_array


def append_kam_phase(one_step_array):
    step_max_index = one_step_array[one_step_array[STEP_PHASE] == 1.].index.max() - 20
    min_index, max_index = np.where(
        (one_step_array[R_PLATE_FORCE_Z] * one_step_array[STEP_PHASE] < -20.)[:step_max_index])[0][[0, -1]]
    mid_index = (min_index + max_index) // 2
    peak_index = one_step_array[R_PLATE_FORCE_Z].loc[:mid_index].idxmin()
    try:
        min_index = np.where(one_step_array.loc[:peak_index, R_KAM_COLUMN] < 0.)[0][-1]
    except IndexError:
        pass
    one_step_array[KAM_PHASE] = [0.] * min_index + [1.] * (max_index - min_index) + [0.] * (
            one_step_array.shape[0] - max_index)
    return one_step_array


def is_step_data_corrupted(one_step_array):
    return (one_step_array[EVENT_COLUMN] >= 0.).all()


def is_foot_on_right_plate_alone(one_step_array):
    min_index, max_index = one_step_array[one_step_array[FORCE_PHASE] == 1.].index[[0, -1]]
    left, right = int(0.7 * min_index + 0.3 * max_index), int(0.3 * min_index + 0.7 * max_index)
    return (one_step_array.loc[left:right, L_PLATE_FORCE_Z] > -20).any()


def is_kam_positive(one_step_array):
    min_index, max_index = one_step_array[one_step_array[FORCE_PHASE] == 1.].index[[0, -1]]
    mid_index = (min_index + max_index) // 2
    peak_index = one_step_array.loc[:mid_index, R_PLATE_FORCE_Z].idxmin()
    return (one_step_array.loc[peak_index:mid_index, R_KAM_COLUMN] > 0.).all()


def is_kam_length_reasonable(one_step_array):
    return np.ptp(np.where(one_step_array[KAM_PHASE] == 1.)) > 30


def keep_stance_data_only(one_step_array):
    one_step_array = one_step_array.loc[one_step_array[FORCE_PHASE] == 1.]
    return one_step_array


def resample_to_100_sample(one_step_array):
    x = np.linspace(0., 1., one_step_array.shape[0])
    new_x = np.linspace(0., 1., 100)
    f = interp1d(x, one_step_array[CONTINUOUS_FIELDS], axis=0, kind=3)
    g = interp1d(x, one_step_array[DISCRETE_FIELDS], axis=0, kind='nearest')
    return pd.DataFrame(np.concatenate([f(new_x), g(new_x)], axis=1), columns=ALL_FIELDS)


def generate_step_data(export_path, processes):
    export_path = os.path.join(DATA_PATH, export_path)
    # create training data and test data
    subject_trial_dict = {}
    for subject_trial, data in all_data_dict.items():
        subject_trial_dict[subject_trial] = get_step_data(data)
        for f in processes:
            subject_trial_dict[subject_trial] = f(subject_trial_dict[subject_trial])

    subject_data_dict = {subject: [data for trial in TRIALS for data in subject_trial_dict[subject + " " + trial]]
                         for subject in SUBJECTS}
    with h5py.File(export_path, 'w') as hf:
        for subject, data_collections in subject_data_dict.items():
            subject_whole_trial = np.concatenate(
                [np.array(data.loc[:, ALL_FIELDS])[np.newaxis, :, :] for data in data_collections], axis=0)
            hf.create_dataset(subject, data=subject_whole_trial, dtype='float32')
        hf.attrs['columns'] = json.dumps(ALL_FIELDS)


if __name__ == "__main__":
    logging.debug("Loading all csv data")
    all_data_dict = {subject + " " + trial: pd.read_csv(os.path.join(DATA_PATH, subject, "combined", trial + ".csv"))
                     for subject in SUBJECTS for trial in TRIALS}
    max_step_length = max([trial_data[EVENT_COLUMN].value_counts().max() for trial_data in all_data_dict.values()])

    SAMPLES_BEFORE_STEP = 40
    PADDING_MODE = PADDING_ZERO
    custom_process = [
        lambda step_data_list: map(append_force_phase, step_data_list),
        lambda step_data_list: map(append_kam_phase, step_data_list),
        lambda step_data_list: filter(is_step_data_corrupted, step_data_list),
        lambda step_data_list: filter(is_foot_on_right_plate_alone, step_data_list),
        lambda step_data_list: filter(is_kam_positive, step_data_list),
        lambda step_data_list: filter(is_kam_length_reasonable, step_data_list)
    ]
    generate_step_data('40samples+stance_swing+padding_zero.h5', custom_process)
    # PADDING_MODE = PADDING_NEXT_STEP
    # generate_step_data('40samples+stance_swing+padding_next_step.h5')
    # PADDING_MODE = PADDING_ZERO
    # TRIALS = ['baseline', 'fpa', 'step_width']
    # generate_step_data('40samples+stance_swing+kick_out_trunksway.h5')
    SAMPLES_BEFORE_STEP = 0
    PADDING_MODE = PADDING_ZERO
    custom_process += [lambda step_data_list: map(keep_stance_data_only, step_data_list),
                       lambda step_data_list: map(resample_to_100_sample, step_data_list)]
    generate_step_data('stance_resampled.h5', custom_process)
