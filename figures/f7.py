from figures.PaperFigures import format_axis
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE, FONT_SIZE_LARGE
import matplotlib.lines as lines
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from figures.f6 import format_errorbar_cap, save_fig
from figures.f6 import print_anova_with_lsd


def draw_sigifi_sign(mean_, std_, bar_locs, one_two=True, two_three=True, one_three=True):
    x_offset = 0.3
    lo = 0.1        # line offset
    y_top = max([a + b for a, b in zip(mean_, std_)])

    for pair, loc_0, loc_1 in zip([one_two, two_three, one_three], [0, 1, 0], [1, 2, 2]):
        if not pair: continue
        if loc_0 == 0 and loc_1 == 2:
            top_line = y_top + 1.4
            coe_0, coe_1 = 0.58, 0.42
            lo = - lo
        else:
            top_line = y_top + 0.5
            coe_0, coe_1 = 0.66, 0.34
        diff_line_0x = [bar_locs[loc_0]+lo, bar_locs[loc_0]+lo, bar_locs[loc_1]-lo, bar_locs[loc_1]-lo]
        diff_line_0y = [mean_[loc_0] + std_[loc_0] + x_offset, top_line, top_line, mean_[loc_1] + std_[loc_1] + x_offset]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[loc_0]*coe_0 + bar_locs[loc_1]*coe_1, top_line+0.15, '*', color='black', size=30)


def draw_f7_bar(_mean, _std, sigifi_sign_fun):
    def format_ticks(x_locs):
        ax = plt.gca()
        ax.set_ylabel('Relative Root Mean Square Error (\%)', fontdict=FONT_DICT_LARGE)
        ax.set_xlabel('KAM Estimation \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ KFM Estimation', fontdict=FONT_DICT_LARGE, labelpad=14)
        ax.set_ylim(0, 14)
        ax.set_yticks(range(0, 15, 2))
        ax.set_yticklabels(range(0, 15, 2), fontdict=FONT_DICT_LARGE)
        ax.set_xlim(-1, 16)
        ax.set_xticks(x_locs)
        ax.set_xticklabels(['', 'IMU \&\nCamera', '', '', 'IMU\nAlone', '', '', 'IMU \&\nCamera', '', '', 'IMU\nAlone', ''], fontdict=FONT_DICT_LARGE, linespacing=0.95)
    rc('text', usetex=True)
    fig = plt.figure(figsize=(9, 9))
    format_axis()
    x_locs = [i + x for x in [0, 4, 9, 13] for i in range(3)]
    # colors = [[0.1, 0.8, 0.8], [0.1, 0.6, 0.1]]
    format_ticks(x_locs)
    bars = []
    for i_condition in range(4):
        bars.append(plt.bar(x_locs[i_condition*3:(i_condition+1)*3], _mean[i_condition*3:(i_condition+1)*3],
                            width=0.7, color=[[0.8, 0.3, 0.3], [0.9, 0.55, 0.55], [1, 0.8, 0.8]]))
        if i_condition == 0:
            sigifi_sign_fun(_mean[i_condition * 3:(i_condition + 1) * 3], _std[i_condition * 3:(i_condition + 1) * 3],
                            x_locs[i_condition * 3:(i_condition + 1) * 3], two_three=False)
        elif i_condition == 1:
            sigifi_sign_fun(_mean[i_condition * 3:(i_condition + 1) * 3], _std[i_condition * 3:(i_condition + 1) * 3],
                            x_locs[i_condition * 3:(i_condition + 1) * 3], one_two=False)
        else:
            sigifi_sign_fun(_mean[i_condition * 3:(i_condition + 1) * 3], _std[i_condition * 3:(i_condition + 1) * 3],
                            x_locs[i_condition * 3:(i_condition + 1) * 3])


    ebar, caplines, barlinecols = plt.errorbar(x_locs, _mean, _std,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 10)
    plt.tight_layout(rect=[0.0, 0.0, 1, 0.88], w_pad=2, h_pad=3)
    plt.legend(bars[0],
               ['All Eight IMUs', 'Three IMUs (Pelvis and both Feet IMUs)', 'One IMU (Pelvis IMU)'],
               # ['All Eight IMUs (\& Cameras)', 'Pelvis, both Feet IMUs (\& camera)', 'Pelvis IMU (\& camera)'],
               handlelength=2, bbox_to_anchor=(0.9, 1.23), ncol=1, fontsize=FONT_SIZE_LARGE, labelspacing=0.2,
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

    print_anova_with_lsd(combo_results[0:3], ['8_fusion_KAM', '3_fusion_KAM', '1_fusion_KAM'])
    print_anova_with_lsd(combo_results[3:6], ['8_alone_KAM', '3_alone_KAM', '1_alone_KAM'])
    print_anova_with_lsd(combo_results[6:9], ['8_fusion_KFM', '3_fusion_KFM', '1_fusion_KFM'])
    print_anova_with_lsd(combo_results[9:], ['8_alone_KFM', '3_alone_KFM', '1_alone_KFM'])

    draw_f7_bar([result.mean() for result in combo_results], [result.sem() for result in combo_results], draw_sigifi_sign)
    save_fig('f7')
    plt.show()


