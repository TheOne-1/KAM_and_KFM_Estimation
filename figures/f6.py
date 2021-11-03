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
import matplotlib.patches as patches
from figures.PaperFigures import save_fig


def format_errorbar_cap(caplines, size=15):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def draw_f6_kam_and_kfm(mean_, std_, sigifi_sign_fun):
    def format_ticks():
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (%)', fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('KAM Estimation                         KFM Estimation', fontdict=FONT_DICT_LARGE, labelpad=13)
        ax.set_ylim(0, 15)
        ax.set_yticks(range(0, 16, 3))
        ax.set_yticklabels(range(0, 16, 3), fontdict=FONT_DICT_LARGE)
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
        ax.set_ylabel('RMSE (% BW·BH)', fontdict=FONT_DICT, labelpad=4)
        ax.set_xlabel('KAM Estimation                              KFM Estimation', fontdict=FONT_DICT, labelpad=13)
        ax.set_ylim(0, 1.2)
        ax.set_yticks(np.arange(0, 1.3, 0.3))
        ax.set_yticklabels(['0', '0.3', '0.6', '0.9', '1.2'], fontdict=FONT_DICT)
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
                               color=np.array([0, 103, 137]) / 255, width=1))
    ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines)
    plt.tight_layout(rect=[0., 0., 1, 1.])
    l2 = lines.Line2D([0.54, 0.54], [0.01, 0.96], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])


def sigifi_sign_8(mean_, std_, bar_locs):
    x_offset = 0.3
    for i, lo in zip([0, 3], [(0.1, 0.1, 0.1), (0.1, 0.1, 0.1)]):
        y_top = max([a + b for a, b in zip(mean_[i:i+3], std_[i:i+3])])
        top_line_y0, top_line_y1 = y_top + 0.6, y_top + 2.1
        """between 1 and 3"""
        diff_line_0x = [bar_locs[i]+lo[0], bar_locs[i]+lo[0], bar_locs[i+1]-lo[1], bar_locs[i+1]-lo[1]]
        diff_line_0y = [mean_[i] + std_[i] + x_offset, top_line_y0, top_line_y0, mean_[i+1] + std_[i+1] + x_offset]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.58 + bar_locs[i+1]*0.42, top_line_y0-0.23, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=40)

        """between 2 and 3"""
        diff_line_0x = [bar_locs[i+1]+lo[1], bar_locs[i+1]+lo[1], bar_locs[i+2]-lo[2], bar_locs[i+2]-lo[2]]
        diff_line_0y = [mean_[i+1] + std_[i+1] + x_offset, top_line_y0, top_line_y0, mean_[i+2] + std_[i+2] + x_offset]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i+1]*0.58 + bar_locs[i+2]*0.42, top_line_y0-0.23, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=40)

        """between 1 and 3"""
        diff_line_1x = [bar_locs[i]-lo[0], bar_locs[i]-lo[0], bar_locs[i+2]+lo[2], bar_locs[i+2]+lo[2]]
        diff_line_1y = [mean_[i] + std_[i] + x_offset, top_line_y1, top_line_y1, mean_[i+2] + std_[i+2] + x_offset]
        plt.plot(diff_line_1x, diff_line_1y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.54 + bar_locs[i+2]*0.46, top_line_y1-0.23, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=40)


def sigifi_sign_8_asb(mean_, std_, bar_locs):
    for i, lo in zip([0, 3], [(0., 0., 0.), (0., 0., 0.)]):
        ax = plt.gca()
        y_top = mean_[1+i] + std_[1+i]
        if i == 0:
            top_line_y0, top_line_y1, top_line_y2 = y_top + 0.1, y_top + 0.2, y_top + 0.2
        else:
            top_line_y0, top_line_y1, top_line_y2 = y_top + 0.1, y_top + 0.2, y_top + 0.3
        diff_line_0x = [bar_locs[i]+lo[0], bar_locs[i+1]-lo[1]]
        diff_line_0y = [top_line_y0, top_line_y0]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH, color=np.array([0, 103, 137]) / 255)
        plt.text(bar_locs[i]*0.59 + bar_locs[i+1]*0.41, top_line_y0-0.07, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=35, zorder=20)
        rect = patches.Rectangle((bar_locs[i]*0.59 + bar_locs[i+1]*0.41, top_line_y0-0.05*y_top), 0.4, 0.075, linewidth=0, color='white', zorder=10)
        ax.add_patch(rect)

        if i == 3:
            diff_line_0x = [bar_locs[i+1]+lo[1], bar_locs[i+2]-lo[2]]
            diff_line_0y = [top_line_y1, top_line_y1]
            plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH, color=np.array([0, 103, 137]) / 255)
            plt.text(bar_locs[i+1]*0.59 + bar_locs[i+2]*0.41, top_line_y1-0.07, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=35, zorder=20)
            rect = patches.Rectangle((bar_locs[i+1]*0.59 + bar_locs[i+2]*0.41, top_line_y1-0.05*y_top), 0.4, 0.075, linewidth=0, color='white', zorder=10)
            ax.add_patch(rect)

        diff_line_1x = [bar_locs[i]-lo[0], bar_locs[i+2]+lo[2]]
        diff_line_1y = [top_line_y2, top_line_y2]
        plt.plot(diff_line_1x, diff_line_1y, 'black', linewidth=LINE_WIDTH, color=np.array([0, 103, 137]) / 255)
        plt.text(bar_locs[i]*0.545 + bar_locs[i+2]*0.455, top_line_y2-0.07, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=35, zorder=20)
        rect = patches.Rectangle((bar_locs[i]*0.545 + bar_locs[i+2]*0.455, top_line_y2-0.05*y_top), 0.4, 0.075, linewidth=0, color='white', zorder=10)
        ax.add_patch(rect)


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
        print('{:.4f} \%'.format(value))


if __name__ == "__main__":
    test_name = '1028'
    target_matric = 'rRMSE_'
    mean_compare_, sem_compare_ = [], []
    combo_results_all = {}
    result_df = pd.read_csv('results/' + test_name + '/estimation_result_individual.csv')
    result_df = result_df[result_df['trial'] == 'all']
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        combo_result_each_moment = []
        for sensor_combo in ['TfnNet', 'Lmf8Imu0Camera', 'Lmf0Imu2Camera']:
            combo_result = result_df[target_matric + sensor_combo + '_' + moment]
            mean_compare_.append(combo_result.mean())
            sem_compare_.append(combo_result.sem())
            combo_result_each_moment.append(combo_result)
            combo_results_all[moment + '_' + sensor_combo] = combo_result
        print_anova_with_lsd(combo_result_each_moment, ['8IMU_2camera', '8IMU', '2camera'])

    """ A KAM & KFM joint figure """
    pd.DataFrame(combo_results_all).to_csv('results/' + test_name + '_anova.csv', index=False)
    print_mean_rrmse(mean_compare_)
    draw_f6_kam_and_kfm(mean_compare_, sem_compare_, sigifi_sign_8)
    save_fig('f6', 600)
    plt.show()
