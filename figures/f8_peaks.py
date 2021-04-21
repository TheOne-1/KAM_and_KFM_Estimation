import pandas as pd
import h5py
import json
from scipy.signal import find_peaks
import numpy as np
from const import LINE_WIDTH, FONT_DICT_SMALL, GRAVITY, FONT_SIZE_LARGE, LINE_WIDTH_THICK, FONT_DICT_LARGE, SUBJECTS
from figures.f6 import save_fig
from figures.PaperFigures import format_axis
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse


def draw_peak(true_peaks, pred_peaks):
    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('KAM: OMC and Force Plates (%BW·BH)', fontdict=FONT_DICT_LARGE, labelpad=12)
        ax.set_ylabel('KAM: Proposed IMU-Camera Fusion Model (%BW·BH)', fontdict=FONT_DICT_LARGE)
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 0.6)
        ax.set_yticks(range(7))      # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ax.set_yticklabels(range(7), fontdict=FONT_DICT_LARGE)
        ax.set_xticks(range(7))      # ['0'] + ['{:3.1f}'.format(x) for x in np.arange(0.1, 0.61, 0.1)]
        ax.set_xticklabels(range(7), fontdict=FONT_DICT_LARGE)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(9, 9))
    blue_dot = plt.scatter(true_peaks, pred_peaks, s=50, marker='.')

    format_axis()
    format_ticks()
    coef = np.polyfit(true_peaks, pred_peaks, 1)
    poly1d_fn = np.poly1d(coef)
    black_line, = plt.plot([0., 6], [0., 6], color='black', linewidth=LINE_WIDTH)
    mean_absolute_error = np.mean(np.abs(np.array(true_peaks) - np.array(pred_peaks)))
    correlation = pearsonr(true_peaks, pred_peaks)[0]

    # plt.text(0.34, 0.02, 'ρ = {:4.2f}\ny = {:4.2f}x + {:4.2f}'.format(
    #     correlation, coef[0], coef[1]), fontdict=FONT_DICT_LARGE)
    plt.text(0.2, 5.5, 'ρ = {:4.2f}\nMAE = {:4.2f} (%BW·BH)'.format(correlation, mean_absolute_error), fontdict=FONT_DICT_LARGE)
    plt.tight_layout(rect=[0, 0, 1, 1])

    save_fig('f8_peak_each_gait_cycle')


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


def get_peaks(data, columns, search_percent_from_start):
    step_lens = get_step_len(data)
    search_lens = (search_percent_from_start * step_lens).astype(int)
    true_row, pred_row = columns.index('true_main_output'), columns.index('pred_main_output')
    true_peaks, pred_peaks = [], []
    peak_not_found = 0
    for i_step in range(data.shape[0]):
        true_peak = find_peak_max(data[i_step, :search_lens[i_step], true_row], 0.1)
        if true_peak is None:
            peak_not_found += 1
            continue
        pred_peak = find_peak_max(data[i_step, :search_lens[i_step], pred_row], 0.1)
        if pred_peak is None:
            pred_peak = np.max(data[i_step, :search_lens[i_step], pred_row])
        true_peaks.append(true_peak / GRAVITY * 100)
        pred_peaks.append(pred_peak / GRAVITY * 100)
    # print('Peaks of {:3.1f}% steps not found.'.format(peak_not_found/data.shape[0]*100))
    return true_peaks, pred_peaks


def get_impulse(data, columns):
    true_row, pred_row = columns.index('true_main_output'), columns.index('pred_main_output')
    true_data = data[:, :, true_row]
    true_data[true_data < 0] = 0
    true_impulse = np.sum(true_data, axis=1) / GRAVITY * 100

    pred_data = data[:, :, pred_row]
    pred_data[pred_data < 0] = 0
    pred_impulse = np.sum(pred_data, axis=1) / GRAVITY * 100

    return true_impulse, pred_impulse


def get_rmse_sample(data, columns):
    force_phase_row = columns.index('force_phase')
    locs = np.where(data[:, :, force_phase_row].ravel())[0]
    true_row, pred_row = columns.index('true_main_output'), columns.index('pred_main_output')
    true_data = data[:, :, true_row].ravel()
    pred_data = data[:, :, pred_row].ravel()
    rmse_sample = np.sqrt(mse(true_data[locs], pred_data[locs])) / GRAVITY * 100
    return rmse_sample


if __name__ == "__main__":
    result_date = 'results/0326'
    with h5py.File(result_date + 'KAM/8IMU_2camera/results.h5', 'r') as hf:
        kam_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kam_data_fields = json.loads(hf.attrs['columns'])
    true_peaks, pred_peaks = get_peaks(kam_data_all_sub[SUBJECTS[16]], kam_data_fields, 0.5)
    draw_peak(true_peaks, pred_peaks)

    with h5py.File(result_date + 'KFM/8IMU_2camera/results.h5', 'r') as hf:
        kfm_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kfm_data_fields = json.loads(hf.attrs['columns'])
    for data, data_fields, sign in zip([kam_data_all_sub, kfm_data_all_sub], [kam_data_fields, kfm_data_fields], [1, -1]):
        for i_trial in range(4):
            true_peak_average, rmse_peak_sub, rmse_sample_sub = [], [], []
            for subject in SUBJECTS:
                sub_data = data[subject]
                sub_trial_data = sub_data[sub_data[:, 0, data_fields.index('trial_id')] == i_trial, :, :]
                sub_trial_data = sub_trial_data * sign
                true_peaks, pred_peaks = get_peaks(sub_trial_data, data_fields, 0.5)
                true_peak_average.append(np.mean(true_peaks))
                rmse_peak_sub.append(np.mean(np.abs((np.array(true_peaks) - np.array(pred_peaks)))))
                rmse_sample_sub.append(get_rmse_sample(sub_trial_data, data_fields))
                # print('{}: {}'.format(subject, get_rmse_sample(sub_trial_data, data_fields)))

            print('{:3.1f} ({:3.1f})\t\t{:4.2f} ({:4.2f})\t\t{:4.2f} ({:4.2f})'.format(
                np.mean(true_peak_average), np.std(true_peak_average),
                np.mean(rmse_peak_sub), np.std(rmse_peak_sub),
                np.mean(rmse_sample_sub), np.std(rmse_sample_sub)))

    plt.show()
