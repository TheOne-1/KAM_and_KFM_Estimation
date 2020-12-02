import os
import pandas as pd
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from customized_logger import logger as logging
from const import TRIALS, SUBJECTS, ALL_FIELDS, PHASE
from const import SAMPLES_BEFORE_STEP, DATA_PATH, LEFT_PLATE_FORCE_Z
from const import PADDING_MODE, PADDING_ZERO, PADDING_NEXT_STEP
from const import R_FORCE_Z_COLUMN, R_KAM_COLUMN, EVENT_COLUMN


def filter_and_clip_data(middle_data, max_len):
    # count the maximum time steps for each gati step
    def form_array_list(array, abnormal_ids):
        target_ids = array[EVENT_COLUMN].drop_duplicates().dropna()
        skip_list = list(target_ids[target_ids < 0])
        if skip_list:
            logging.debug('Containing corrupted data in steps: {}. These will be dropped out'.format(skip_list))
        for _id in target_ids:
            if -abs(_id) in skip_list:
                continue
            # fetch data before a step
            stance_swing_step = array[array[EVENT_COLUMN] == _id]
            step_begin = stance_swing_step.index.min() - SAMPLES_BEFORE_STEP
            if PADDING_MODE == PADDING_ZERO:
                step_end = stance_swing_step.index.max()
            elif PADDING_MODE == PADDING_NEXT_STEP:
                step_end = stance_swing_step.index.min() + max_len
            else:
                raise RuntimeError("PADDING_MODE is not appropriate.")
            if step_begin < array.index.min():
                continue
            expected_step = array.loc[step_begin:step_end, :].copy()
            if (expected_step[EVENT_COLUMN] < 0).any():
                continue
            # -20: swing phase might contain stance phase of next step, in which case, force might be positive.
            stance_swing_step_cliped = stance_swing_step.iloc[:-20]
            stance_phase = stance_swing_step_cliped[stance_swing_step_cliped[R_FORCE_Z_COLUMN] < -20]
            stance_phase_min_index = stance_phase.index.min()
            stance_phase_max_index = stance_phase.index.max()
            stance_phase_mid_index = (stance_phase_min_index + stance_phase_max_index) // 2
            stance_phase_peak_index = stance_phase.loc[:stance_phase_mid_index, R_FORCE_Z_COLUMN].idxmin()
            stance_phase = stance_swing_step_cliped.loc[stance_phase_min_index:stance_phase_max_index]
            if np.mean(stance_phase.loc[
                       int(0.7 * stance_phase_min_index + 0.3 * stance_phase_max_index):
                       int(0.3 * stance_phase_min_index + 0.7 * stance_phase_max_index), LEFT_PLATE_FORCE_Z]) < -20:
                if is_verbose:
                    plt.figure()
                    plt.plot(stance_phase[LEFT_PLATE_FORCE_Z].values)
                    plt.show()
                continue
            if (stance_phase.loc[stance_phase_peak_index:stance_phase_mid_index, R_KAM_COLUMN] < 0.).any():
                abnormal_ids.append(_id)
                if is_verbose:
                    plt.figure()
                    plt.plot(stance_swing_step[R_KAM_COLUMN].values)
                    plt.show()
                continue
            kam_keep_begin = stance_phase[stance_phase[R_KAM_COLUMN] < 0.].loc[:stance_phase_peak_index].index.max()
            if np.isnan(kam_keep_begin):
                kam_keep_begin = stance_phase[R_KAM_COLUMN].loc[:stance_phase_peak_index].idxmin()
                assert stance_phase_min_index <= kam_keep_begin <= stance_phase_max_index
            kam_keep_end = stance_phase_max_index
            if kam_keep_end - kam_keep_begin < 30:
                continue
            expected_step[PHASE] = 0.
            expected_step.loc[kam_keep_begin:kam_keep_end, PHASE] = 1.
            expected_step.index = range(expected_step.shape[0])
            expected_step = expected_step.reindex(range(max_len + SAMPLES_BEFORE_STEP))
            expected_step[np.isnan(expected_step)] = 0
            yield expected_step

    ab_ids = []
    step_list = list(form_array_list(middle_data, ab_ids))
    return step_list, ab_ids


def generate_subject_data():
    # create training data and test data
    logging.debug("Loading all csv data")
    all_data_dict = {subject + " " + trial: pd.read_csv(os.path.join(DATA_PATH, subject, "combined", trial + ".csv"))
                     for subject in SUBJECTS for trial in TRIALS}
    max_step_length = max([trial_data[EVENT_COLUMN].value_counts().max() for trial_data in all_data_dict.values()])
    subject_trial_dict = {}
    for subject_trial, data in all_data_dict.items():
        logging.debug("{} is now in filter and clip process".format(subject_trial))
        subject_trial_dict[subject_trial], abnormal_steps = filter_and_clip_data(data, max_step_length)
        if abnormal_steps:
            logging.warning(
                "In {}, Total count of abnormal gait ids with KAM is negative is: {}.The details is {}.".format(
                    subject_trial, len(abnormal_steps), abnormal_steps))
    subject_dict = {subject: [data for trial in TRIALS for data in subject_trial_dict[subject + " " + trial]]
                    for subject in SUBJECTS}
    return subject_dict


def generate_step_data(export_path):
    export_path = os.path.join(DATA_PATH, export_path)
    subject_data_dict = generate_subject_data()
    with h5py.File(export_path, 'w') as hf:
        for subject, data_collections in subject_data_dict.items():
            subject_whole_trial = np.concatenate(
                [np.array(data.loc[:, ALL_FIELDS])[np.newaxis, :, :] for data in data_collections], axis=0)
            hf.create_dataset(subject, data=subject_whole_trial, dtype='float32')
        hf.attrs['columns'] = json.dumps(ALL_FIELDS)


if __name__ == "__main__":
    is_verbose = False
    PADDING_MODE = PADDING_ZERO
    generate_step_data('40samples+stance_swing+padding_zero.h5')
    # PADDING_MODE = PADDING_NEXT_STEP
    # generate_step_data('40samples+stance_swing+padding_next_step.h5')
    # PADDING_MODE = PADDING_ZERO
    # TRIALS = ['baseline', 'fpa', 'step_width']
    # generate_step_data('40samples+stance_swing+kick_out_trunksway.h5')
