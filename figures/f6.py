from PaperFigures import format_axis
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE, SENSOR_COMBINATION
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from scipy.stats import ttest_rel


def draw_f6(mean_sel, std_sel):
    def format_errorbar_cap(caplines):
        for i_cap in range(1):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(15)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    def format_ticks():
        ax = plt.gca()
        ax.set_xlabel('KAM Estimation \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ KFM Estimation', fontdict=FONT_DICT_LARGE, labelpad=12)
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        ax.set_yticklabels(range(0, 11, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xticks([0.5, 3.5, 6.5, 10.5, 13.5, 16.5])
        ax.set_xticklabels(['IMU \&\nCamera', 'IMU', 'Camera', 'IMU \&\nCamera', 'IMU', 'Camera'], fontdict=FONT_DICT_LARGE)

    rc('text', usetex=True)
    fig = plt.figure(figsize=(9, 9))
    # format_axis()
    # format_ticks()
    sel_locs = [0, 3, 6, 9, 12]
    bar_patterns = ['', '', '', '', '']

    for i_condition in range(5):
        bar_sel = plt.bar(sel_locs[i_condition], mean_sel[i_condition], hatch=bar_patterns[i_condition % 5],
                          color=[0.8, 0.1, 0.1], width=1)
    ebar, caplines, barlinecols = plt.errorbar(sel_locs, mean_sel, std_sel,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    # format_errorbar_cap(caplines)
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.96], w_pad=2, h_pad=3)
    # plt.legend([bar_sel, bar_all],
    #            ['Selected Features from IMU and/or Camera', 'All the Features from IMU and/or Camera'],
    #            handlelength=3, bbox_to_anchor=(0.2, 0.95), ncol=1, fontsize=FONT_SIZE,
    #            frameon=False, labelspacing=0.2)


def save_fig():
    plt.savefig('exports/f6.png')


if __name__ == "__main__":
    target_matric = 'RMSE_'
    mean_sel, std_sel, mean_all, std_all = [], [], [], []
    for i_moment, moment in enumerate(['KAM']):
        sel_feature_df = pd.read_csv('results/0127_' + moment + '/estimation_result_individual.csv')        # 0127_all_feature_
        sel_feature_df = sel_feature_df[sel_feature_df['trial'] == 'all']
        for sensor_combo in ['8IMU_2camera', '8IMU', '3IMU_2camera', '3IMU', '2camera']:
            sel_feature_result = sel_feature_df[target_matric + sensor_combo]
            mean_sel.append(sel_feature_result.mean())
            std_sel.append(sel_feature_result.std())
    draw_f6(mean_sel, std_sel)
    # save_fig()
    plt.show()
