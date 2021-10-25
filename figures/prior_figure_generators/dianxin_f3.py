# -*- coding: utf-8 -*-
import os
import h5py
import json
from figures.PaperFigures import get_mean_std, format_axis
from const import SUBJECTS
import numpy as np
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
FONT_DICT = {'fontsize': FONT_SIZE, 'fontname': 'Microsoft YaHei'}


def draw_f3(mean_std_kam, mean_std_kfm):
    def draw_subplot(ax, mean_std):
        arr_true_mean, arr_true_std, arr_pred_mean, arr_pred_std = mean_std['true_mean'], mean_std['true_std'], \
                                                                   mean_std['pred_mean'], mean_std['pred_std']
        axis_x = range(arr_true_mean.shape[0])
        ax.plot(axis_x, arr_true_mean, color='green', label='真值', linewidth=LINE_WIDTH*2)
        ax.fill_between(axis_x, arr_true_mean - arr_true_std, arr_true_mean + arr_true_std,
                        facecolor='green', alpha=0.3)
        ax.plot(axis_x, arr_pred_mean, color='peru', label='估计值', linewidth=LINE_WIDTH*2)
        ax.fill_between(axis_x, arr_pred_mean - arr_pred_std, arr_pred_mean + arr_pred_std,
                        facecolor='peru', alpha=0.3)
        ax.tick_params(labelsize=FONT_DICT['fontsize'])
        ax.set_xticks(range(0, 101, 25))
        ax.set_xticklabels(range(0, 101, 25), fontdict=FONT_DICT)
        ax.set_xlabel('支撑相 (%)', fontdict=FONT_DICT)
        ax.set_xlim(0, 100)
        ax.legend(handlelength=3, loc='upper right', ncol=1, fontsize=FONT_SIZE,
               frameon=False)
        format_axis()

    def subplot_1_style():
        ax = plt.gca()
        ax.set_ylabel('膝关节内翻力矩 (BW X BH)', fontdict=FONT_DICT)
        ax.set_ylim(-0.21, 0.41)
        ticks = [-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT)

    def subplot_2_style():
        ax = plt.gca()
        ax.set_ylabel('膝关节屈曲力矩 (BW X BH)', fontdict=FONT_DICT)
        ax.set_ylim(-0.3, 0.80)
        ticks = [-0.2, 0., 0.2, 0.4, 0.6, 0.8]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontdict=FONT_DICT)

    rc('text', usetex=False)
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 6])        # , width_ratios=[8, 1, 8]
    draw_subplot(fig.add_subplot(gs[1, 0]), mean_std_kam)
    subplot_1_style()
    draw_subplot(fig.add_subplot(gs[1, 1]), mean_std_kfm)
    subplot_2_style()
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99], w_pad=3)
    plt.savefig('exports/dianxin_f3.png')


if __name__ == "__main__":
    with h5py.File('results/KAM_12sub/results.h5', 'r') as hf:
        kam_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kam_data_fields = json.loads(hf.attrs['columns'])
    with h5py.File('results/KFM_12sub/results.h5', 'r') as hf:
        kfm_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kfm_data_fields = json.loads(hf.attrs['columns'])

    kam_mean_std = get_mean_std(np.concatenate(list(kam_data_all_sub.values()), axis=0), kam_data_fields, 'main_output')
    kfm_mean_std = get_mean_std(np.concatenate(list(kfm_data_all_sub.values()), axis=0), kfm_data_fields, 'main_output')
    kfm_mean_std['true_mean'], kfm_mean_std['pred_mean'] = -kfm_mean_std['true_mean'], -kfm_mean_std['pred_mean']
    draw_f3(kam_mean_std, kfm_mean_std)
    plt.show()
