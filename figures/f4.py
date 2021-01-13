from PaperFigures import get_mean_std, format_plot, get_trial_data
from const import SUBJECTS
import numpy as np
import pandas as pd
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE, TRIALS, TRIALS_PRINT
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc


def init_figure():
    rc('text', usetex=True)
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=[1, 5, 5])        # , width_ratios=[8, 1, 8]
    return fig, gs


def draw_subplot(ax, trial_index, mean_std_IMU_OP, mean_std_IMU):
    arr_true_mean = mean_std_IMU_OP['true_mean']
    arr_pred_mean_IMU_OP = mean_std_IMU_OP['pred_mean']
    arr_pred_mean_IMU = mean_std_IMU['pred_mean']

    axis_x = range(arr_true_mean.shape[0])
    ax.plot(axis_x, arr_true_mean, color='green', label='Laboratory Force Plate \& Optical Motion Capture', linewidth=LINE_WIDTH*1)
    ax.plot(axis_x, arr_pred_mean_IMU_OP, color='peru', label='Portable IMU \& Smartphone Camera', linewidth=LINE_WIDTH*1)
    ax.plot(axis_x, arr_pred_mean_IMU, '--', color='peru', label='Portable IMU', linewidth=LINE_WIDTH*1)
    ax.tick_params(labelsize=FONT_DICT['fontsize'])
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels(range(0, 101, 25), fontdict=FONT_DICT)
    ax.set_xlabel('Stance Phase (\%)', fontdict=FONT_DICT)
    ax.set_xlim(0, 100)
    format_plot()
    if moment == 'KAM':
        plt.title(TRIALS_PRINT[trial_index], fontsize=FONT_SIZE)
        kam_subplot_style()
    elif moment == 'KFM':
        kfm_subplot_style()


def kam_subplot_style():
    ax = plt.gca()
    ax.set_ylabel('KAM (BW X BH)', fontdict=FONT_DICT)
    ax.set_ylim(-0.16, 0.31)
    ticks = [-0.15, 0., 0.15, 0.3]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontdict=FONT_DICT)


def kfm_subplot_style():
    ax = plt.gca()
    ax.set_ylabel('KFM (BW X BH)', fontdict=FONT_DICT)
    ax.set_ylim(-0.31, 0.61)
    ticks = [-0.3, 0., 0.3, 0.6]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontdict=FONT_DICT)


def save_fig():
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99], w_pad=2)
    plt.legend(handlelength=3, bbox_to_anchor=(-0.1, 3.2), ncol=2, fontsize=FONT_SIZE,
               frameon=False)
    plt.savefig('exports/f4.png')


if __name__ == "__main__":
    fig, gs = init_figure()
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        for trial_index, trial_name in enumerate(TRIALS):
            _data_IMU_OP, _data_fields = get_trial_data('results/0107_' + moment + '/IMU+OP/results.h5', trial_index)
            mean_std_IMU_OP = get_mean_std(np.concatenate(_data_IMU_OP, axis=0), _data_fields, 'main_output')
            _data_IMU, _data_fields = get_trial_data('results/0107_' + moment + '/IMU/results.h5', trial_index)
            mean_std_IMU = get_mean_std(np.concatenate(_data_IMU, axis=0), _data_fields, 'main_output')
            if moment == 'KFM':
                mean_std_IMU_OP['true_mean'], mean_std_IMU_OP['pred_mean'] = -mean_std_IMU_OP['true_mean'], -mean_std_IMU_OP['pred_mean']
                mean_std_IMU['true_mean'], mean_std_IMU['pred_mean'] = -mean_std_IMU['true_mean'], -mean_std_IMU['pred_mean']
            draw_subplot(fig.add_subplot(gs[i_moment+1, trial_index]), trial_index, mean_std_IMU_OP, mean_std_IMU)
    save_fig()
    plt.show()










