from PaperFigures import format_axis
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE, SENSOR_COMBINATION
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.stats import ttest_rel
from scikit_posthocs import posthoc_tukey, posthoc_ttest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import prettytable as pt
import numpy as np


SENSOR_COMBINATION_SORTED = ['8IMU_2camera', '3IMU_2camera', '8IMU', '3IMU', '2camera']


def format_errorbar_cap(caplines):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(15)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def draw_f6(mean_sel, std_sel, title):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
        ax.set_ylim(0, 9)
        ax.set_yticks(range(0, 10, 1))
        ax.set_yticklabels(range(0, 10, 1), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 9)
        ax.set_xticks(range(0, 9, 2))
        ax.set_xticklabels(['8 IMUs \&\n2 Cameras', '3 IMUs \&\n2 Cameras', '8 IMUs',
                            '3 IMUs', '2 Cameras'], fontdict=FONT_DICT_LARGE)

    rc('text', usetex=True)
    fig = plt.figure(figsize=(9, 9))
    format_axis()
    format_ticks()
    sel_locs = range(0, 9, 2)
    # bar_colors = ['red', 'orange', 'green', 'blue', 'slategrey']

    for i_condition in range(5):
        bar_sel = plt.bar(sel_locs[i_condition], mean_sel[i_condition],
                          color=[0.8, 0.3, 0.3], width=0.6)
    ebar, caplines, barlinecols = plt.errorbar(sel_locs, mean_sel, std_sel,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96], w_pad=2, h_pad=3)


def draw_f6_kam_and_kfm(mean_, std_, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('KAM Estimation \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ KFM Estimation', fontdict=FONT_DICT_LARGE, labelpad=12)
        ax.set_ylim(0, 12)
        ax.set_yticks(range(0, 13, 2))
        ax.set_yticklabels(range(0, 13, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 12)
        ax.set_xticks([0, 2, 4, 7, 9, 11])
        ax.set_xticklabels(['IMU \&\n Camera', 'IMU', 'Camera', 'IMU \&\n Camera', 'IMU', 'Camera'], fontdict=FONT_DICT_LARGE)

    rc('text', usetex=True)
    fig = plt.figure(figsize=(9, 9))
    format_axis()
    format_ticks()
    bar_locs = [0, 2, 4, 7, 9, 11]
    sigifi_sign_fun(mean_, std_, bar_locs)
    bar_ = []
    for i_condition in range(6):
        bar_.append(plt.bar(bar_locs[i_condition], mean_[i_condition],
                               color=[0.8, 0.3, 0.3], width=1))
    ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96], w_pad=2, h_pad=3)
    # plt.legend([bar_[0], bar_[-1]],
    #            ['KAM estimation', 'KFM estimation'],
    #            handlelength=3, bbox_to_anchor=(0.3, 0.95), ncol=1, fontsize=FONT_SIZE,
    #            frameon=False, labelspacing=0.2)
    l2 = lines.Line2D([0.54, 0.54], [0.01, 0.9], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])


def sigifi_sign_8(mean_, std_, bar_locs):
    offset = 0.3
    diff_line_0x = [bar_locs[0], bar_locs[0], bar_locs[1], bar_locs[1]]
    diff_line_0y = [mean_[0] + std_[0] + offset, 10.3, 10.3, mean_[1] + std_[1] + offset]
    plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
    plt.text(bar_locs[0]*0.55 + bar_locs[1]*0.45, 10.5, '*', color='black', size=40)
    diff_line_1x = [bar_locs[0], bar_locs[0], bar_locs[2], bar_locs[2]]
    diff_line_1y = [mean_[0] + std_[0] + offset, 11.5, 11.5, mean_[2] + std_[2] + offset]
    plt.plot(diff_line_1x, diff_line_1y, 'black', linewidth=LINE_WIDTH)
    plt.text(bar_locs[0]*0.55 + bar_locs[2]*0.45, 11.7, '*', color='black', size=40)

    diff_line_0x = [bar_locs[4], bar_locs[4], bar_locs[5], bar_locs[5]]
    diff_line_0y = [mean_[4] + std_[4] + offset, 10.3, 10.3, mean_[5] + std_[5] + offset]
    plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
    plt.text(bar_locs[4]*0.55 + bar_locs[5]*0.45, 10.5, '*', color='black', size=40)
    diff_line_1x = [bar_locs[3], bar_locs[3], bar_locs[5], bar_locs[5]]
    diff_line_1y = [mean_[3] + std_[3] + offset, 11.5, 11.5, mean_[5] + std_[5] + offset]
    plt.plot(diff_line_1x, diff_line_1y, 'black', linewidth=LINE_WIDTH)
    plt.text(bar_locs[3]*0.55 + bar_locs[5]*0.45, 11.7, '*', color='black', size=40)


def print_paired_t_test(combo_results):
    tb = pt.PrettyTable()
    tb.field_names = SENSOR_COMBINATION_SORTED
    for i_combo, combo_a in enumerate(SENSOR_COMBINATION_SORTED):
        p_val_row = []
        for j_combo, combo_b in enumerate(SENSOR_COMBINATION_SORTED):
            p_val = round(ttest_rel(combo_results[i_combo], combo_results[j_combo]).pvalue, 2)
            if np.isnan(p_val):
                p_val = 1
            p_val_row.append(p_val)
        tb.add_row(p_val_row)
    tb.add_column('', SENSOR_COMBINATION_SORTED, align="l")
    print(tb)


def print_pairwise_tukeyhsd(combo_results):
    endog = np.array(combo_results)
    groups = [name for name in SENSOR_COMBINATION_SORTED for i in range(len(combo_results[0]))]
    results = pairwise_tukeyhsd(endog.ravel(), groups)
    print(results)


def save_fig(name):
    plt.savefig('exports/' + name + '.png')


if __name__ == "__main__":
    target_matric = 'rRMSE_'
    mean_compare_8, sem_compare_8, mean_compare_3, sem_compare_3 = [], [], [], []
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        test_df = pd.read_csv('results/0131_all_feature_' + moment + '/estimation_result_individual.csv')        # 0127_all_feature_ 0127_selected_feature_
        test_df = test_df[test_df['trial'] == 'all']
        combo_results = []
        mean_moment, sem_moment = [], []

        for sensor_combo in SENSOR_COMBINATION_SORTED:
            combo_result = test_df[target_matric + sensor_combo]
            combo_results.append(combo_result)
        print('\nP values for {} estimation'.format(moment))
        print_paired_t_test(combo_results)

        for sensor_combo in ['8IMU_2camera', '8IMU', '2camera']:
            combo_result = test_df[target_matric + sensor_combo]
            mean_compare_8.append(combo_result.mean())
            sem_compare_8.append(combo_result.sem())

    """ A KAM & KFM joint figure """
    draw_f6_kam_and_kfm(mean_compare_8, sem_compare_8, sigifi_sign_8)
    save_fig('f6')
    plt.show()
