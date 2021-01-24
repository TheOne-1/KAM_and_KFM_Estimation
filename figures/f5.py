from PaperFigures import format_axis
from const import SUBJECTS
import numpy as np
import matplotlib.patches as patches
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd


def draw_f5(mean_sel, std_sel, mean_all, std_all):

    def format_errorbar_cap(caplines):
        for i_cap in range(1):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(15)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('KAM Estimation \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ KFM Estimation', fontdict=FONT_DICT, labelpad=12)
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT)
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        ax.set_yticklabels(range(0, 11, 2), fontdict=FONT_DICT)
        ax.set_xticks([0.5, 3.5, 6.5, 10.5, 13.5, 16.5])
        ax.set_xticklabels(['IMU \&\nCamera', 'IMU', 'Camera', 'IMU \&\nCamera', 'IMU', 'Camera'], fontdict=FONT_DICT)

    rc('text', usetex=True)
    fig = plt.figure(figsize=(9, 9))
    format_axis()
    format_ticks()
    sel_locs, all_locs = [0, 3, 6, 10, 13, 16], [1, 4, 7, 11, 14, 17]
    bar_patterns = ['', '', '']

    for i_condition in range(6):
        bar_sel = plt.bar(sel_locs[i_condition], mean_sel[i_condition], hatch=bar_patterns[i_condition % 3],
                color=[0.8, 0.1, 0.1], width=1)
        bar_all = plt.bar(all_locs[i_condition], mean_all[i_condition], hatch=bar_patterns[i_condition % 3],
                color=[0.8, 0.5, 0.5], width=1)
    ebar, caplines, barlinecols = plt.errorbar(sel_locs + all_locs, mean_sel + mean_all, std_sel + std_all,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96], w_pad=2, h_pad=3)
    plt.legend([bar_sel, bar_all],
               ['Selected Features for the Proposed Model', 'All the Features from IMU and/or Camera'],
               handlelength=3, bbox_to_anchor=(0.2, 0.95), ncol=1, fontsize=FONT_SIZE,
               frameon=False, labelspacing=0.2)
    l2 = lines.Line2D([0.53, 0.53], [0.01, 0.9], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])


def save_fig():
    plt.savefig('exports/f5.png')


if __name__ == "__main__":
    target_matric = 'rRMSE_'
    mean_sel, std_sel, mean_all, std_all = [], [], [], []
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        sel_feature_df = pd.read_csv('results/0122_' + moment + '/estimation_result_individual.csv')
        sel_feature_df = sel_feature_df[sel_feature_df['trial'] == 'all']
        all_feature_df = pd.read_csv('results/0122_used_all_the_features_' + moment + '/estimation_result_individual.csv')
        all_feature_df = all_feature_df[all_feature_df['trial'] == 'all']
        for data_modal in ['IMU_OP', 'IMU', 'OP']:
            sel_feature_result = sel_feature_df[target_matric + data_modal]
            mean_sel.append(sel_feature_result.mean())
            std_sel.append(sel_feature_result.std())
            all_feature_result = all_feature_df[target_matric + data_modal]
            mean_all.append(all_feature_result.mean())
            std_all.append(sel_feature_result.std())
    draw_f5(mean_sel, std_sel, mean_all, std_all)
    save_fig()
    plt.show()
