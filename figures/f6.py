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


def format_ticks():
    ax = plt.gca()
    ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 12, 1))
    ax.set_yticklabels(range(0, 12, 1), fontdict=FONT_DICT_LARGE)
    ax.set_xlim(-1, 9)
    ax.set_xticks(range(0, 9, 2))
    ax.set_xticklabels(['8 IMUs \&\n2 Cameras', '3 IMUs \&\n2 Cameras', '8 IMUs',
                        '3 IMUs', '2 Cameras'], fontdict=FONT_DICT_LARGE)
    # ax.set_xticklabels(['8 IMUs \&\n2 Cameras', '3 IMUs \&\n2 Cameras', '8 IMUs',
    #                     '1 IMUs \&\n2 Cameras', '3 IMUs', '2 Cameras', '1 IMUs'], fontdict=FONT_DICT_LARGE)


def draw_f6(mean_sel, std_sel, title):
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
    # plt.legend([bar_sel, bar_all],
    #            ['Selected Features from IMU and/or Camera', 'All the Features from IMU and/or Camera'],
    #            handlelength=3, bbox_to_anchor=(0.2, 0.95), ncol=1, fontsize=FONT_SIZE,
    #            frameon=False, labelspacing=0.2)


def draw_f6_kam_and_kfm(mean_sel, std_sel):
    rc('text', usetex=True)
    fig = plt.figure(figsize=(13, 9))
    format_axis()
    format_ticks()
    sel_locs = [i - 0.3 for i in range(0, 13, 2)]
    sel_locs.extend([i + 0.3 for i in range(0, 13, 2)])
    # sel_locs.sort()
    bar_colors = [[0.8, 0.1, 0.1], [0.8, 0.6, 0.6]]
    bar_sel = []
    for i_condition in range(14):
        bar_sel.append(plt.bar(sel_locs[i_condition], mean_sel[i_condition],
                               color=bar_colors[int(i_condition > 6)], width=0.6))
    ebar, caplines, barlinecols = plt.errorbar(sel_locs, mean_sel, std_sel,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96], w_pad=2, h_pad=3)
    plt.legend([bar_sel[0], bar_sel[-1]],
               ['KAM estimation', 'KFM estimation'],
               handlelength=3, bbox_to_anchor=(0.3, 0.95), ncol=1, fontsize=FONT_SIZE,
               frameon=False, labelspacing=0.2)


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
    mean_all, std_all = [], []
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        test_df = pd.read_csv('results/0127_selected_feature_' + moment + '/estimation_result_individual.csv')        # 0127_all_feature_ 0127_selected_feature_
        test_df = test_df[test_df['trial'] == 'all']
        combo_results = []
        mean_moment, std_moment = [], []
        for sensor_combo in SENSOR_COMBINATION_SORTED:
            combo_result = test_df[target_matric + sensor_combo]
            mean_moment.append(combo_result.mean())
            std_moment.append(combo_result.std())
            combo_results.append(combo_result)

            mean_all.append(combo_result.mean())
            std_all.append(combo_result.std())

        print('\nP values for {} estimation'.format(moment))
        print_paired_t_test(combo_results)
        draw_f6(mean_moment, std_moment, moment)
        save_fig(moment)

    """ A KAM & KFM joint figure """
    # draw_f6_kam_and_kfm(mean_all, std_all)
    save_fig('both KAM and KFM')
    plt.show()
