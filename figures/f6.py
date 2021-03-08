from figures.PaperFigures import format_axis
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE, SENSOR_COMBINATION_SORTED
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import prettytable as pt
import numpy as np


def format_errorbar_cap(caplines, size=15):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def draw_f6_various_imus(_mean, _std, title):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
        ax.set_ylim(0, 12)
        ax.set_yticks(range(0, 13, 2))
        ax.set_yticklabels(range(0, 13, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 13)
        ax.set_xticks(range(0, 13, 2))
        ax.set_xticklabels(['8 IMUs \&\n2 Cameras', '3 IMUs \&\n2 Cameras', '8 IMUs', '1 IMU \&\n2 Cameras',
                            '3 IMUs', '2 Cameras', '1 IMU'], fontdict=FONT_DICT_LARGE)

    rc('text', usetex=True)
    fig = plt.figure(figsize=(13, 7))
    format_axis()
    format_ticks()
    sel_locs = range(0, 13, 2)
    for i_condition in range(7):
        _bars = plt.bar(sel_locs[i_condition], _mean[i_condition], color=[0.8, 0.3, 0.3], width=0.6)
    ebar, caplines, barlinecols = plt.errorbar(sel_locs, _mean, _std,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96], w_pad=2, h_pad=3)
    plt.grid()
    plt.title(title, fontsize=30)


def draw_f6_kam_and_kfm(mean_, std_, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('KAM Estimation \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ KFM Estimation', fontdict=FONT_DICT_LARGE, labelpad=14)
        ax.set_ylim(0, 12)
        ax.set_yticks(range(0, 13, 2))
        ax.set_yticklabels(range(0, 13, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 12)
        ax.set_xticks([0, 2, 4, 7, 9, 11])
        ax.set_xticklabels(['IMU \&\n Camera', 'IMU \n Alone', 'Camera \n Alone', 'IMU \&\n Camera', 'IMU \n Alone', 'Camera \n Alone'], fontdict=FONT_DICT_LARGE)

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
    plt.tight_layout(rect=[0., 0., 1, 1], w_pad=2, h_pad=3)
    # plt.legend([bar_[0], bar_[-1]],
    #            ['KAM estimation', 'KFM estimation'],
    #            handlelength=3, bbox_to_anchor=(0.3, 0.95), ncol=1, fontsize=FONT_SIZE,
    #            frameon=False, labelspacing=0.2)
    l2 = lines.Line2D([0.54, 0.54], [0.01, 0.96], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])


def draw_f6_for_ISB(mean_, std_, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE, labelpad=4)
        ax.set_xlabel('\ \ \ KAM Estimation \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ KFM Estimation', fontdict=FONT_DICT_LARGE, linespacing=0.95, labelpad=12)
        ax.set_ylim(0, 12)
        ax.set_yticks(range(0, 13, 2))
        ax.set_yticklabels(range(0, 13, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 12)
        ax.set_xticks([0, 2, 4, 7, 9, 11])
        ax.set_xticklabels(['IMU \&\n Camera', 'IMU \n Alone', 'Camera \n Alone', 'IMU \&\n Camera', 'IMU \n Alone', 'Camera \n Alone'], fontdict=FONT_DICT_LARGE, linespacing=0.95)

    rc('text', usetex=True)
    fig = plt.figure(figsize=(11, 6.2))
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
    plt.tight_layout(rect=[0., -0.02, 1, 1.04], w_pad=2, h_pad=3)
    l2 = lines.Line2D([0.54, 0.54], [0.01, 0.96], linestyle='--', transform=fig.transFigure, color='gray')
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


def print_anova_with_lsd(combo_results, combo_names=SENSOR_COMBINATION_SORTED):
    """ Equals to doing paired t-tests. """
    tb = pt.PrettyTable()
    tb.field_names = combo_names
    for i_combo, combo_a in enumerate(combo_names):
        p_val_row = []
        for j_combo, combo_b in enumerate(combo_names):
            if i_combo == j_combo:
                p_val = 1
            else:
                p_val = round(ttest_rel(combo_results[i_combo], combo_results[j_combo]).pvalue, 3)
            p_val_row.append(p_val)
        tb.add_row(p_val_row)
    tb.add_column('', combo_names, align="l")
    print(tb)


def print_pairwise_tukeyhsd(combo_results, combo_names):
    endog = np.array(combo_results)
    groups = [name for name in combo_names for i in range(len(combo_results[0]))]
    results = pairwise_tukeyhsd(endog.ravel(), groups)
    print(results)


def print_mean_rrmse(mean_values):
    for value in mean_values:
        print('{:.1f} \%'.format(value))


def save_fig(name):
    plt.savefig('exports/' + name + '.png')


if __name__ == "__main__":
    test_name = '0307'
    target_matric = 'rRMSE_'
    mean_compare_, sem_compare_ = [], []
    combo_results_all = {}
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        test_df = pd.read_csv('results/' + test_name + moment + '/estimation_result_individual.csv')
        test_df = test_df[test_df['trial'] == 'all']

        combo_result_each_moment = []
        for sensor_combo in ['8IMU_2camera', '8IMU', '2camera']:
            combo_result = test_df[target_matric + sensor_combo]
            mean_compare_.append(combo_result.mean())
            sem_compare_.append(combo_result.sem())
            combo_result_each_moment.append(combo_result)
            combo_results_all[moment + '_' + sensor_combo] = combo_result
        print_anova_with_lsd(combo_result_each_moment, ['8IMU_2camera', '8IMU', '2camera'])

    """ A KAM & KFM joint figure """
    pd.DataFrame(combo_results_all).to_csv('results/' + test_name + '_anova.csv', index=False)
    print_mean_rrmse(mean_compare_)
    # draw_f6_for_ISB(mean_compare_8, sem_compare_8, sigifi_sign_8)
    draw_f6_kam_and_kfm(mean_compare_, sem_compare_, sigifi_sign_8)
    save_fig('f6')
    plt.show()
