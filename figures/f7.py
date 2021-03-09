from figures.PaperFigures import format_axis
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE, FONT_SIZE_LARGE
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import prettytable as pt
import numpy as np
from figures.f6 import format_errorbar_cap, save_fig


def draw_f7_bar(_mean, _std):
    def format_ticks(x_locs):
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('KAM Estimation \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ KFM Estimation', fontdict=FONT_DICT_LARGE, labelpad=14)
        ax.set_ylim(0, 12)
        ax.set_yticks(range(0, 13, 2))
        ax.set_yticklabels(range(0, 13, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 16)
        # x_locs = [i + x for x in [0, 4, 9, 13] for i in range(3)]
        ax.set_xticks(x_locs)
        ax.set_xticklabels(['', 'IMU \&\nCamera', '', '', 'IMU\nAlone', '', '', 'IMU \&\nCamera', '', '', 'IMU\nAlone', ''], fontdict=FONT_DICT_LARGE)
    rc('text', usetex=True)
    fig = plt.figure(figsize=(9, 9))
    format_axis()
    x_locs = [i + x for x in [0, 4, 9, 13] for i in range(3)]
    # colors = [[0.1, 0.8, 0.8], [0.1, 0.6, 0.1]]
    format_ticks(x_locs)
    bars = []
    for i_condition in range(4):
        i_moment = int(np.floor(i_condition / 2))
        i_camera_used = int(i_condition % 2)
        bars.append(plt.bar(x_locs[i_condition*3:(i_condition+1)*3], _mean[i_condition*3:(i_condition+1)*3],
                            # width=0.7, color=[[0.1, 0.3, 0.3], [0.1, 0.6, 0.6], [0.1, 0.9, 0.9]]))
                            width=0.7, color=[[0.8, 0.3, 0.3], [0.9, 0.55, 0.55], [1, 0.8, 0.8]]))

    ebar, caplines, barlinecols = plt.errorbar(x_locs, _mean, _std,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 10)
    plt.tight_layout(rect=[0.0, 0.0, 1, 0.88], w_pad=2, h_pad=3)
    plt.legend(bars[0],
               ['All Eight IMUs', 'Pelvis and both Feet IMUs', 'Pelvis IMU'],
               # ['All Eight IMUs (\& Cameras)', 'Pelvis, both Feet IMUs (\& camera)', 'Pelvis IMU (\& camera)'],
               handlelength=2, bbox_to_anchor=(0.7, 1.25), ncol=1, fontsize=FONT_SIZE_LARGE, labelspacing=0.2,
               frameon=False)
    l2 = lines.Line2D([0.54, 0.54], [0.01, 0.8], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l2])


if __name__ == "__main__":
    target_matric = 'rRMSE_'
    combo_results = []
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        test_df = pd.read_csv(
            'results/0307' + moment + '/estimation_result_individual.csv')
        test_df = test_df[test_df['trial'] == 'all']

        """ Figure showing results of 8, 3, and 1 IMU for discussion """
        for sensor_combo in ['8IMU_2camera', '3IMU_2camera', '1IMU_2camera', '8IMU', '3IMU', '1IMU']:
            combo_result = test_df[target_matric + sensor_combo]
            combo_results.append(combo_result)

    draw_f7_bar([result.mean() for result in combo_results], [result.sem() for result in combo_results])
    save_fig('f7')
    plt.show()


