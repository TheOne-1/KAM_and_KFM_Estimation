import numpy as np
import matplotlib.pyplot as plt
from const import LINE_WIDTH, TRIALS
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE, SUBJECTS
from base_framework import BaseFramework
import h5py
import json
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from const import GRAVITY
from scipy.signal import find_peaks


def get_step_len(data, feature_col_num=0):
    """
    :param data: Numpy array, 3d (step, sample, feature)
    :param feature_col_num: int, feature column id for step length detection. Different id would probably return
           the same results
    :return:
    """
    data_the_feature = data[:, :, feature_col_num]
    zero_loc = data_the_feature == 0.
    step_lens = np.sum(~zero_loc, axis=1)
    return step_lens


def find_peak_max(data_clip, height, width=None, prominence=None):
    """
    find the maximum peak
    :return:
    """
    peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
    if len(peaks) == 0:
        return None
    peak_heights = properties['peak_heights']
    return np.max(peak_heights)


def get_mean_gait_cycle_then_find_peak(data, columns, search_percent_from_start):
    mean_std = get_mean_std(data, columns, 'main_output')
    search_sample = int(100 * search_percent_from_start)
    true_peak = find_peak_max(mean_std['true_mean'][:search_sample], 0.1)
    pred_peak =find_peak_max(mean_std['pred_mean'][:search_sample], 0.1)
    if true_peak is None:
        true_peak = 0
    return true_peak, pred_peak


def get_peak_of_each_gait_cycle(data, columns, search_percent_from_start):
    step_lens = get_step_len(data)
    search_lens = (search_percent_from_start * step_lens).astype(int)
    true_row, pred_row = columns.index('true_main_output'), columns.index('pred_main_output')
    true_peaks, pred_peaks = [], []
    peak_not_found = 0
    for i_step in range(data.shape[0]):
        true_peak = np.max(data[i_step, :search_lens[i_step], true_row])
        pred_peak = np.max(data[i_step, :search_lens[i_step], pred_row])
        # true_peak = find_peak_max(data[i_step, :search_lens[i_step], true_row], 0.1)
        # if true_peak is None:
        #     peak_not_found += 1
        #     continue
        # pred_peak = find_peak_max(data[i_step, :search_lens[i_step], pred_row], 0.1)
        # if pred_peak is None:
        #     pred_peak = np.max(data[i_step, :search_lens[i_step], pred_row])
        true_peaks.append(true_peak / GRAVITY * 100)
        pred_peaks.append(pred_peak / GRAVITY * 100)
    # print('Peaks of {:3.1f}% steps not found.'.format(peak_not_found/data.shape[0]*100))
    return true_peaks, pred_peaks


def format_axis(line_width=LINE_WIDTH):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=line_width)
    ax.yaxis.set_tick_params(width=line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def hide_axis_add_grid():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color='lightgray', linewidth=1.5)
    ax.tick_params(color='lightgray', width=1.5)


def get_mean_std(data_array, data_fields, col_name, transfer_to_newton=True):
    true_index, pred_index = data_fields.index('true_' + col_name), data_fields.index('pred_' + col_name)
    weight_index = data_fields.index(FORCE_PHASE)

    true_stance, _ = BaseFramework.keep_stance_then_resample(data_array[:, :, true_index:true_index + 1], data_array[:, :, weight_index:weight_index + 1], 101)
    pred_stance, _ = BaseFramework.keep_stance_then_resample(data_array[:, :, pred_index:pred_index + 1], data_array[:, :, weight_index:weight_index + 1], 101)
    true_mean, true_std = true_stance[:, :, 0].mean(axis=0), true_stance[:, :, 0].std(axis=0)
    pred_mean, pred_std = pred_stance[:, :, 0].mean(axis=0), pred_stance[:, :, 0].std(axis=0)
    if transfer_to_newton:
        true_mean, true_std, pred_mean, pred_std = true_mean / GRAVITY * 100, true_std / GRAVITY * 100, pred_mean / GRAVITY * 100, pred_std / GRAVITY * 100
    return {'true_mean': true_mean, 'true_std': true_std, 'pred_mean': pred_mean, 'pred_std': pred_std}


def get_data(file_path, specific_trial=None, subjects=SUBJECTS):
    with h5py.File(file_path, 'r') as hf:
        _data_all = {subject: subject_data[:] for subject, subject_data in hf.items() if subject in subjects}
        _data_fields = json.loads(hf.attrs['columns'])

    if specific_trial is not None:
        trial_id_col_loc = _data_fields.index('trial_id')
        _data_all = [data[data[:, 0, trial_id_col_loc] == specific_trial, :, :] for data in _data_all.values()]
    return _data_all, _data_fields


def get_score(arr_true, arr_pred, w):
    assert(len(arr_true.shape) == 1 and arr_true.shape == arr_pred.shape == w.shape)
    locs = np.where(w.ravel())[0]
    arr_true, arr_pred = arr_true.ravel()[locs], arr_pred.ravel()[locs]
    mae = np.mean(np.abs(arr_true - arr_pred)) / GRAVITY * 100
    r_rmse = np.sqrt(mse(arr_true, arr_pred)) / (arr_true.max() - arr_true.min()) * 100
    cor_value = pearsonr(arr_true, arr_pred)[0]
    rmse = np.sqrt(mse(arr_true, arr_pred)) / GRAVITY * 100  # TODO: error-prone code here. Modify generate_combined_data to calculate external KAM.
    return {'MAE': mae, 'RMSE': rmse, 'rRMSE': r_rmse, 'r':  cor_value}


def get_gait_params(data_file, export_file):
    _data_all, _data_fields = get_data(data_file)
    marker_index = {marker: [_data_fields.index(marker + axis) for axis in ['X', 'Y', 'Z']]
                    for marker in ['RFM2_', 'RFCC_', 'CV7_', 'LIPS_', 'RIPS_', 'LFCC_']}
    with h5py.File(export_file, 'w') as hf:
        for subject in SUBJECTS:
            sub_data = _data_all[subject]
            step_lens = get_step_len(sub_data)
            fpa = get_fpa(sub_data[:, :, marker_index['RFM2_']], sub_data[:, :, marker_index['RFCC_']], step_lens)
            ts = get_trunk_sway(sub_data[:, :, marker_index['CV7_']], sub_data[:, :, marker_index['LIPS_']],
                                sub_data[:, :, marker_index['RIPS_']], step_lens)
            sw = get_step_width(sub_data[:, :, marker_index['LFCC_'][0]], sub_data[:, :, marker_index['RFCC_'][0]], step_lens)
            subject_id = np.full(fpa.shape, SUBJECTS.index(subject))
            trial_id = sub_data[:, 0, _data_fields.index('trial_id')]
            sub_gait_data = np.column_stack([subject_id, trial_id, fpa, sw, ts])
            hf.create_dataset(subject, data=sub_gait_data, dtype='float32')
        hf.attrs['columns'] = json.dumps(['subject_id', 'trial_id'] + TRIALS[1:])


def get_fpa(toe, heel, step_lens):
    forward_vector = toe - heel
    fpa_sample = 180 / np.pi * np.arctan2(forward_vector[:, :, 0], forward_vector[:, :, 1])
    start_samples = ((step_lens - 40) * 0.15 + 20).astype(int)
    end_samples = ((step_lens - 40) * 0.5 + 20).astype(int)
    fpa_step = np.zeros(step_lens.shape)
    for i_step in range(toe.shape[0]):
        fpa_step[i_step] = np.mean(fpa_sample[i_step, start_samples[i_step]:end_samples[i_step]])
    return fpa_step


def get_trunk_sway(c7, l_pelvis, r_pelvis, step_lens):
    trunk_vector = c7 - (l_pelvis + r_pelvis) / 2
    trunk_angle_sample = - 180 / np.pi * np.arctan(trunk_vector[:, :, 0] / trunk_vector[:, :, 2])
    trunk_angle = np.zeros(step_lens.shape)
    for i_step in range(c7.shape[0]):
        trunk_angle[i_step] = np.max(trunk_angle_sample[i_step, :step_lens[i_step]]) \
                              - np.min(trunk_angle_sample[i_step, :step_lens[i_step]])
    return trunk_angle


def get_step_width(l_heel_x, r_heel_x, step_lens):
    r_foot_loc = r_heel_x[:, 40]
    step_width = np.zeros(step_lens.shape)
    for i_step in range(l_heel_x.shape[0]):
        step_width[i_step] = r_foot_loc[i_step] - l_heel_x[i_step, step_lens[i_step]]
    return step_width















