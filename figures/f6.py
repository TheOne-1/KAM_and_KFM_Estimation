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
        ax.set_xticklabels(['8 IMUs &\n2 Cameras', '3 IMUs &\n2 Cameras', '8 IMUs', '1 IMU &\n2 Cameras',
                            '3 IMUs', '2 Cameras', '1 IMU'], fontdict=FONT_DICT_LARGE)

    rc('text', usetex=True)
    fig = plt.figure(figsize=(13, 7))
    format_axis()
    format_ticks()
    sel_locs = range(0, 13, 2)
    for i_condition in range(7):
        _bars = plt.bar(sel_locs[i_condition], _mean[i_condition], color=[0, 103, 137], width=0.6)
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
        ax.set_ylabel('Relative Root Mean Square Error (%)', fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('KAM Estimation                         KFM Estimation', fontdict=FONT_DICT_LARGE, labelpad=13)
        ax.set_ylim(0, 12)
        ax.set_yticks(range(0, 13, 3))
        ax.set_yticklabels(range(0, 13, 3), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 11.5)
        ax.set_xticks([0, 2, 4, 6.5, 8.5, 10.5])
        ax.set_xticklabels(['IMUs &\nCameras', 'IMUs\nAlone', 'Cameras\nAlone', 'IMUs &\nCameras',
                            'IMUs \nAlone', 'Cameras\nAlone'], fontdict=FONT_DICT_LARGE, linespacing=0.95)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(10, 7))
    format_axis()
    format_ticks()
    bar_locs = [0, 2, 4, 6.5, 8.5, 10.5]
    sigifi_sign_fun(mean_, std_, bar_locs)
    bar_ = []
    for i_condition in range(6):
        bar_.append(plt.bar(bar_locs[i_condition], mean_[i_condition],
                               color=np.array([0, 103, 137]) / 255, width=1))
    ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 20)
    plt.tight_layout(rect=[0., -0.01, 1, 1.01], w_pad=2, h_pad=3)
    l2 = lines.Line2D([0.54, 0.54], [0.01, 0.96], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])


def draw_f6_for_ASB(mean_, std_, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (%)', fontdict=FONT_DICT, labelpad=4)
        ax.set_xlabel('KAM Estimation                              KFM Estimation', fontdict=FONT_DICT, labelpad=13)
        ax.set_ylim(0, 12)
        ax.set_yticks(range(0, 13, 2))
        ax.set_yticklabels(range(0, 13, 2), fontdict=FONT_DICT)
        ax.set_xlim(-1, 12)
        ax.set_xticks([0, 2, 4, 7, 9, 11])
        ax.set_xticklabels(['IMU &\n Camera', 'IMU \n Alone', 'Camera \n Alone', 'IMU &\n Camera', 'IMU \n Alone', 'Camera \n Alone'], fontdict=FONT_DICT, linespacing=0.95)

    rc('font', family='Arial')
    fig = plt.figure(figsize=(9, 5.9))
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
    plt.tight_layout(rect=[0., -0.02, 1, 1.01], w_pad=2, h_pad=3)
    l2 = lines.Line2D([0.54, 0.54], [0.01, 0.96], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])


def sigifi_sign_8(mean_, std_, bar_locs):
    x_offset = 0.3
    # star_offset =
    for i, lo in zip([0, 3], [(0.1, 0., 0.), (0.1, 0.1, 0.1)]):
        y_top = max([a + b for a, b in zip(mean_[i:i+3], std_[i:i+3])])
        top_line_y0, top_line_y1 = y_top + 0.8, y_top + 2
        diff_line_0x = [bar_locs[i]+lo[0], bar_locs[i]+lo[0], bar_locs[i+1]-lo[1], bar_locs[i+1]-lo[1]]
        diff_line_0y = [mean_[i] + std_[i] + x_offset, top_line_y0, top_line_y0, mean_[i+1] + std_[i+1] + x_offset]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.58 + bar_locs[i+1]*0.42, top_line_y0-0.35, '*', color='black', size=50)

        if i == 3:
            diff_line_0x = [bar_locs[i+1]+lo[1], bar_locs[i+1]+lo[1], bar_locs[i+2]-lo[2], bar_locs[i+2]-lo[2]]
            diff_line_0y = [mean_[i+1] + std_[i+1] + x_offset, top_line_y0, top_line_y0, mean_[i+2] + std_[i+2] + x_offset]
            plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
            plt.text(bar_locs[i+1]*0.58 + bar_locs[i+2]*0.42, top_line_y0-0.35, '*', color='black', size=50)

        diff_line_1x = [bar_locs[i]-lo[0], bar_locs[i]-lo[0], bar_locs[i+2]+lo[2], bar_locs[i+2]+lo[2]]
        diff_line_1y = [mean_[i] + std_[i] + x_offset, top_line_y1, top_line_y1, mean_[i+2] + std_[i+2] + x_offset]
        plt.plot(diff_line_1x, diff_line_1y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.54 + bar_locs[i+2]*0.46, top_line_y1-0.35, '*', color='black', size=50)


def sigifi_sign_8_asb(mean_, std_, bar_locs):
    x_offset = 0.3
    # star_offset =
    for i, lo in zip([0, 3], [(0.1, 0., 0.), (0.1, 0.1, 0.1)]):
        y_top = max([a + b for a, b in zip(mean_[i:i+3], std_[i:i+3])])
        top_line_y0, top_line_y1 = y_top + 0.8, y_top + 2
        diff_line_0x = [bar_locs[i]+lo[0], bar_locs[i]+lo[0], bar_locs[i+1]-lo[1], bar_locs[i+1]-lo[1]]
        diff_line_0y = [mean_[i] + std_[i] + x_offset, top_line_y0, top_line_y0, mean_[i+1] + std_[i+1] + x_offset]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.59 + bar_locs[i+1]*0.41, top_line_y0-0.5, '*', color='black', size=50)

        if i == 3:
            diff_line_0x = [bar_locs[i+1]+lo[1], bar_locs[i+1]+lo[1], bar_locs[i+2]-lo[2], bar_locs[i+2]-lo[2]]
            diff_line_0y = [mean_[i+1] + std_[i+1] + x_offset, top_line_y0, top_line_y0, mean_[i+2] + std_[i+2] + x_offset]
            plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
            plt.text(bar_locs[i+1]*0.59 + bar_locs[i+2]*0.41, top_line_y0-0.5, '*', color='black', size=50)

        diff_line_1x = [bar_locs[i]-lo[0], bar_locs[i]-lo[0], bar_locs[i+2]+lo[2], bar_locs[i+2]+lo[2]]
        diff_line_1y = [mean_[i] + std_[i] + x_offset, top_line_y1, top_line_y1, mean_[i+2] + std_[i+2] + x_offset]
        plt.plot(diff_line_1x, diff_line_1y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.545 + bar_locs[i+2]*0.455, top_line_y1-0.5, '*', color='black', size=50)


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


def save_fig(name, dpi=300):
    plt.savefig('exports/' + name + '.png', dpi=dpi)


if __name__ == "__main__":
    test_name = '0326'
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
    draw_f6_for_ASB(mean_compare_, sem_compare_, sigifi_sign_8_asb)
    # draw_f6_kam_and_kfm(mean_compare_, sem_compare_, sigifi_sign_8)
    save_fig('f6', 600)
    plt.show()
