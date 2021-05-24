from figures.PaperFigures import get_mean_std, hide_axis_add_grid, get_data
from const import SUBJECTS
import numpy as np
import matplotlib.patches as patches
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE, TRIALS, TRIALS_PRINT
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
from figures.f6 import save_fig


def init_figure():
    rc('text', usetex=True)
    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=[1, 15, 15])  # , width_ratios=[8, 1, 8]
    return fig, gs


def draw_subplot(ax, trial_index, mean_std_IMU_OP):
    arr_true_mean = mean_std_IMU_OP['true_mean']
    arr_pred_mean_IMU_OP = mean_std_IMU_OP['pred_mean']
    color_0, color_1 = np.array([255, 166, 0]) / 255, np.array([0, 103, 137]) / 255
    axis_x = range(arr_true_mean.shape[0])
    ax.plot(axis_x, arr_true_mean, color=color_0, label='Laboratory Force Plate \& Optical Motion Capture',
            linewidth=LINE_WIDTH * 1)
    ax.plot(axis_x, arr_pred_mean_IMU_OP, '--', color=color_1, label='Portable IMU \& Smartphone Camera',
            linewidth=LINE_WIDTH * 1)
    # ax.plot(axis_x, arr_pred_mean_IMU, '--', color='peru', label='Portable IMU', linewidth=LINE_WIDTH * 1)
    ax.tick_params(labelsize=FONT_DICT['fontsize'])
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels(['0\%', '', '50\%', '', '100\%'], fontdict=FONT_DICT)
    ax.set_xlim(-1, 101)
    hide_axis_add_grid()
    plt.title(TRIALS_PRINT[trial_index], fontsize=FONT_SIZE)
    if moment == 'KAM':
        kam_subplot_style(trial_index)
    elif moment == 'KFM':
        kfm_subplot_style(trial_index)


def kam_subplot_style(trial_index):
    ax = plt.gca()
    if trial_index == 0:
        ax.set_ylabel('KAM (\%BW$\cdot$BH)', fontdict=FONT_DICT)
    ax.set_ylim(-2.1, 4.1)
    ax.set_yticks([-2, -0.5, 1, 2.5, 4])
    ax.set_yticklabels([-2, '', 1, '', 4], fontdict=FONT_DICT)


def kfm_subplot_style(trial_index):
    ax = plt.gca()
    if trial_index == 0:
        ax.set_ylabel('KFM (\%BW$\cdot$BH)', fontdict=FONT_DICT)
    ax.set_ylim(-3.1, 9.1)
    ax.set_yticks([-3, 0., 3, 6, 9])
    ax.set_yticklabels([-3, '', 3, '', 9], fontdict=FONT_DICT)


def organize_fig():
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99], w_pad=2, h_pad=3)
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((-485, -6), 605, 19, fill=False, clip_on=False, ec='gray'))
    ax.add_patch(patches.Rectangle((-485, 14), 605, 19, fill=False, clip_on=False, ec='gray'))
    plt.legend(handlelength=3, bbox_to_anchor=(-4.6, 2.9), ncol=1, fontsize=FONT_SIZE,
               frameon=False, labelspacing=0.2)


if __name__ == "__main__":
    fig, gs = init_figure()
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        for trial_index, trial_name in enumerate(TRIALS):
            _data_IMU_OP, _data_fields = get_data('results/0326' + moment + '/8IMU_2camera/results.h5', trial_index, SUBJECTS[15])
            mean_std_IMU_OP = get_mean_std(np.concatenate(_data_IMU_OP, axis=0), _data_fields, 'main_output')
            if moment == 'KFM':
                mean_std_IMU_OP['true_mean'], mean_std_IMU_OP['pred_mean'] = -mean_std_IMU_OP['true_mean'], - \
                mean_std_IMU_OP['pred_mean']
            draw_subplot(fig.add_subplot(gs[i_moment + 1, trial_index]), trial_index, mean_std_IMU_OP)
    organize_fig()
    save_fig('f4')
    plt.show()
