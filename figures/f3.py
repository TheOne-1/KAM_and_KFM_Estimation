import os
import h5py
import json
from PaperFigures import get_mean_std, format_plot
from const import SUBJECTS
import numpy as np
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc


def draw_f3(mean_std_kam, mean_std_kfm):
    def draw_subplot(ax, mean_std):
        arr_true_mean, arr_true_std, arr_pred_mean, arr_pred_std = mean_std
        axis_x = range(arr_true_mean.shape[0])
        ax.plot(axis_x, arr_true_mean, color='green', label='Portable IMU \& Smartphone Camera', linewidth=LINE_WIDTH*2)
        ax.fill_between(axis_x, arr_true_mean - arr_true_std, arr_true_mean + arr_true_std,
                        facecolor='green', alpha=0.3)
        ax.plot(axis_x, arr_pred_mean, color='peru', label='Laboratory Force Plate \& Optical Motion Capture', linewidth=LINE_WIDTH*2)
        ax.fill_between(axis_x, arr_pred_mean - arr_pred_std, arr_pred_mean + arr_pred_std,
                        facecolor='peru', alpha=0.3)
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_xticks(range(0, 101, 25))
        ax.set_xticklabels(range(0, 101, 25), fontdict=FONT_DICT)
        ax.set_xlabel('Stance Phase (\%)', fontdict=FONT_DICT)
        ax.set_xlim(0, 100)

        # ax.set_yticks(range(-30, 41, 10))
        # y_tick_list = ['-30', '-20', '-10', '0', '10', '20', '30', '40']
        # ax.set_yticklabels(y_tick_list, fontdict=FONT_DICT)
        format_plot()

    def subplot_1_style():
        ax = plt.gca()
        ax.set_ylabel('Knee Adduction Moment (BW X BH)', fontdict=FONT_DICT)
        ax.set_ylim(-0.21, 0.41)
        ticks = [-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT)

    def subplot_2_style():
        ax = plt.gca()
        ax.set_ylabel('Knee Flexion Moment (BW X BH)', fontdict=FONT_DICT)
        ax.set_ylim(-0.21, 0.41)
        ticks = [-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT)

    fig = plt.figure(figsize=(7, 13))
    gs = gridspec.GridSpec(nrows=2, ncols=1)        # , width_ratios=[8, 1, 8]
    draw_subplot(fig.add_subplot(gs[0, 0]), mean_std_kam)
    subplot_1_style()
    plt.legend(handlelength=3, bbox_to_anchor=(1.1, 1.23), ncol=1, fontsize=FONT_SIZE,
               frameon=False)
    draw_subplot(fig.add_subplot(gs[1, 0]), mean_std_kfm)
    subplot_2_style()
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
    plt.savefig('exports/f3.png')


if __name__ == "__main__":
    result_dir = './results/0107_KAM/IMU+OP'
    with h5py.File(os.path.join(result_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])
    rc('text', usetex=True)

    sub_mean_std_then_mean_list = [get_mean_std(_data_all_sub[sub], _data_fields, 'main_output') for sub in SUBJECTS]
    sub_mean_std_then_mean = np.mean(np.array(sub_mean_std_then_mean_list), axis=0)
    all_mean_std = get_mean_std(np.concatenate(list(_data_all_sub.values()), axis=0), _data_fields, 'main_output')
    draw_f3(all_mean_std, sub_mean_std_then_mean)
    plt.show()
