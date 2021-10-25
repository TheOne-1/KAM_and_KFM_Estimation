from PaperFigures import get_mean_std, hide_axis_add_grid, get_trial_data
from const import SUBJECTS
import numpy as np
import matplotlib.patches as patches
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE, TRIALS, TRIALS_PRINT
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc

def init_figure():
    rc('text', usetex=True)
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=[1, 5, 5])  # , width_ratios=[8, 1, 8]
    return fig, gs


def draw_subplot(ax, trial_index, mean_std_IMU):
    arr_true_mean = mean_std_IMU['true_mean']
    arr_pred_mean_IMU = mean_std_IMU['pred_mean']

    axis_x = range(arr_true_mean.shape[0])
    ax.plot(axis_x, arr_true_mean, color='green', label='True Value',
            linewidth=LINE_WIDTH * 1)
    ax.plot(axis_x, arr_pred_mean_IMU, '--', color='peru', label='Estimated Value', linewidth=LINE_WIDTH * 1)
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
        ax.set_ylabel('KAM (BW X BH)', fontdict=FONT_DICT)
    ax.set_ylim(-0.21, 0.41)
    ax.set_yticks([-0.2, -0.05, 0.1, 0.25, 0.4])
    ax.set_yticklabels([-0.2, '', 0.1, '', 0.4], fontdict=FONT_DICT)


def kfm_subplot_style(trial_index):
    ax = plt.gca()
    if trial_index == 0:
        ax.set_ylabel('KFM (BW X BH)', fontdict=FONT_DICT)
    ax.set_ylim(-0.31, 0.91)
    ax.set_yticks([-0.3, 0., 0.3, 0.6, 0.9])
    ax.set_yticklabels([-0.3, '', 0.3, '', 0.9], fontdict=FONT_DICT)


def organize_fig():
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99], w_pad=2, h_pad=3)
    ax = plt.gca()
    ax.add_patch(patches.Rectangle((-496, -0.6), 616, 1.8, fill=False, clip_on=False, ec='gray'))
    ax.add_patch(patches.Rectangle((-496, 1.3), 616, 1.8, fill=False, clip_on=False, ec='gray'))
    plt.legend(handlelength=3, bbox_to_anchor=(-1.5, 3.5), ncol=1, fontsize=FONT_SIZE,
               frameon=False, labelspacing=0.2)


def save_fig():
    plt.savefig('exports/f4.png')


if __name__ == "__main__":
    fig, gs = init_figure()
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        for trial_index, trial_name in enumerate(TRIALS):
            _data_IMU, _data_fields = get_trial_data('results/' + moment + '_12sub/results.h5', trial_index,
                                                     SUBJECTS[1])
            mean_std_IMU = get_mean_std(np.concatenate(_data_IMU, axis=0), _data_fields, 'main_output')
            if moment == 'KFM':
                mean_std_IMU['true_mean'], mean_std_IMU['pred_mean'] = -mean_std_IMU['true_mean'], -mean_std_IMU[
                    'pred_mean']
            draw_subplot(fig.add_subplot(gs[i_moment + 1, trial_index]), trial_index, mean_std_IMU)
    organize_fig()
    save_fig()
    plt.show()
