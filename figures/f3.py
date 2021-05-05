import os
import h5py
import json
from figures.PaperFigures import get_mean_std, format_axis
from const import SUBJECTS
import numpy as np
from const import LINE_WIDTH, FONT_DICT_SMALL, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from figures.f6 import save_fig
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc


def draw_f3(mean_std_kam, mean_std_kfm):
    def draw_subplot(ax, mean_std):
        arr_true_mean, arr_true_std, arr_pred_mean, arr_pred_std = mean_std['true_mean'], mean_std['true_std'], \
                                                                   mean_std['pred_mean'], mean_std['pred_std']
        axis_x = range(arr_true_mean.shape[0])
        color_0, color_1 = np.array([255, 166, 0]) / 255, np.array([0, 103, 137]) / 255
        ax.plot(axis_x, arr_true_mean, color=color_0, label='Laboratory Force Plate & Optical Motion Capture', linewidth=LINE_WIDTH*2)
        ax.fill_between(axis_x, arr_true_mean - arr_true_std, arr_true_mean + arr_true_std,
                        facecolor=color_0, alpha=0.4)
        ax.plot(axis_x, arr_pred_mean, '--', color=color_1, label='Portable IMUs & Smartphone Cameras', linewidth=LINE_WIDTH*2)
        ax.fill_between(axis_x, arr_pred_mean - arr_pred_std, arr_pred_mean + arr_pred_std,
                        facecolor=color_1, alpha=0.4)
        ax.tick_params(labelsize=FONT_DICT_SMALL['fontsize'])
        ax.set_xticks(range(0, 101, 25))
        ax.set_xticklabels(range(0, 101, 25), fontdict=FONT_DICT_SMALL)
        ax.set_xlabel('Stance Phase (%)', fontdict=FONT_DICT_SMALL)
        ax.set_xlim(0, 100)
        format_axis()

    def subplot_1_style():
        ax = plt.gca()
        ax.set_ylabel('Knee Adduction Moment (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ax.set_ylim(-2, 4)
        ticks = [-2, 0, 2, 4]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT_SMALL)

    def subplot_2_style():
        ax = plt.gca()
        ax.set_ylabel('Knee Flexion Moment (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ax.set_ylim(-2.5, 7.5)
        ticks = [-2.5, 0, 2.5, 5, 7.5]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(16, 5.6))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6])
    draw_subplot(fig.add_subplot(gs[1, 0]), mean_std_kam)
    subplot_1_style()
    draw_subplot(fig.add_subplot(gs[1, 1]), mean_std_kfm)
    subplot_2_style()
    plt.tight_layout(rect=[0., -0.02, 1., 1.1], w_pad=3)
    plt.legend(handlelength=3, bbox_to_anchor=(-0.22, 1.22), ncol=1, fontsize=FONT_DICT_SMALL['fontsize'],
               frameon=False)
    save_fig('f3', 600)


if __name__ == "__main__":
    with h5py.File('results/0326KAM/8IMU_2camera/results.h5', 'r') as hf:
        kam_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kam_data_fields = json.loads(hf.attrs['columns'])
    with h5py.File('results/0326KFM/8IMU_2camera/results.h5', 'r') as hf:
        kfm_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kfm_data_fields = json.loads(hf.attrs['columns'])

    kam_mean_std = get_mean_std(np.concatenate(list(kam_data_all_sub.values()), axis=0), kam_data_fields, 'main_output')
    kfm_mean_std = get_mean_std(np.concatenate(list(kfm_data_all_sub.values()), axis=0), kfm_data_fields, 'main_output')
    kfm_mean_std['true_mean'], kfm_mean_std['pred_mean'] = -kfm_mean_std['true_mean'], -kfm_mean_std['pred_mean']
    draw_f3(kam_mean_std, kfm_mean_std)
    plt.show()
