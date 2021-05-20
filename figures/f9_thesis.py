import h5py
import json
import pandas as pd
import numpy as np
from const import LINE_WIDTH, GRAVITY, FONT_DICT_SMALL, SUBJECTS, TRIALS
from figures.f6 import save_fig
from figures.PaperFigures import get_peak_of_each_gait_cycle, format_axis, get_mean_gait_cycle_then_find_peak
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as lines
from more_itertools import sort_together
from scipy.stats import ttest_rel
import matplotlib.gridspec as gridspec


def draw_sigifi_sign(mean_, std_, bar_locs, p_between_pattern, ylim):
    one_two, two_three, one_three = p_between_pattern
    lo = [0. for i in range(3)]        # line offset
    for i_bar, there_are_two_lines in enumerate([one_two and one_three, one_two and two_three, two_three and one_three]):
        if there_are_two_lines:
            lo[i_bar] = 0.1
    y_top = max([a + b for a, b in zip(mean_, std_)])

    for pair, loc_0, loc_1 in zip([one_two, two_three, one_three], [0, 1, 0], [1, 2, 2]):
        if not pair: continue
        if loc_0 == 0 and loc_1 == 2:
            lo_sign = -1
            if not one_two and not two_three:
                top_line = y_top + 0.07 * ylim
            else:
                top_line = y_top + 0.22 * ylim
            coe_0, coe_1 = 0.53, 0.47
        else:
            lo_sign = 1
            top_line = y_top + 0.08 * ylim
            coe_0, coe_1 = 0.56, 0.44
        diff_line_0x = [bar_locs[loc_0]+lo_sign*lo[loc_0], bar_locs[loc_0]+lo_sign*lo[loc_0], bar_locs[loc_1]-lo_sign*lo[loc_1], bar_locs[loc_1]-lo_sign*lo[loc_1]]
        diff_line_0y = [mean_[loc_0] + std_[loc_0] + 0.04 * ylim, top_line, top_line, mean_[loc_1] + std_[loc_1] + 0.04 * ylim]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text((bar_locs[loc_0]+lo_sign*lo[loc_0])*coe_0 + (bar_locs[loc_1]-lo_sign*lo[loc_1])*coe_1, top_line - 0.06 * ylim, '*', color='black', size=40)


def format_errorbar_cap(caplines, size=15):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def sort_gait_cycles_according_to_param(param_config, sub_param, sub_data):
    ts_id = TRIALS.index(param_config['trial_name'])
    sub_ts_all = sub_param[sub_param[:, gait_param_fields.index('trial_id')] == ts_id, gait_param_fields.index(param_config['trial_name'])]
    sub_trial_data = sub_data[sub_data[:, 0, data_fields.index('trial_id')] == ts_id, :, :]
    param_sorted, sub_trial_data_sorted = sort_together((sub_ts_all, sub_trial_data))
    return sub_trial_data_sorted, param_sorted


def init_f9():
    rc('font', family='Arial')
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5, 5])        # , width_ratios=[8, 1, 8]
    return fig, gs


def draw_f9_subplot(mean_, std_, p_between_pattern, ax, moment_name):
    def format_kam_y_ticks():
        ax.set_ylabel('Peak KAM (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ylim = 6
        ax.set_ylim(0, ylim)
        ax.set_yticks(range(0, ylim+1, 2))
        ax.set_yticklabels(range(0, ylim+1, 2), fontdict=FONT_DICT_SMALL)
        return ylim

    def format_kfm_y_ticks():
        ax.set_ylabel('Peak KFM (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ylim = 10
        ax.set_ylim(0, ylim)
        ax.set_yticks(range(0, ylim+1, 2))
        ax.set_yticklabels(range(0, ylim+1, 2), fontdict=FONT_DICT_SMALL)
        return ylim

    def format_x_ticks():
        ax.set_xlabel('Foot Progression Angle                    Step Width             '
                      '             Trunk Sway Angle   ', fontdict=FONT_DICT_SMALL, labelpad=5)
        ax.set_xlim(-1, 26)
        ax.set_xticks(np.arange(0.5, 25, 3))
        ax.set_xticklabels(['Toe-in', 'Normal', 'Toe-out', 'Narrow', 'Normal', 'Wide', 'Small', 'Medium', 'Large'],
                           fontdict=FONT_DICT_SMALL, linespacing=0.95)

    format_axis()
    format_x_ticks()
    bar_locs = [x + y for x in range(0, 27, 3) for y in [0, 1]]
    color_0, color_1 = np.array([255, 166, 0]) / 255, np.array([0, 103, 137]) / 255
    colors = [color_0, color_1] * int(len(bar_locs) / 2)
    bar_ = []
    for i_condition in range(len(bar_locs)):
        bar_.append(plt.bar(bar_locs[i_condition], mean_[i_condition], color=colors[i_condition], width=1))
    ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 20)
    if moment_name == 'KAM':
        ylim = format_kam_y_ticks()
        plt.legend(bar_[0:2], ['Laboratory Force Plate & Optical Motion Capture', 'Portable IMUs & Smartphone Cameras'],
            handlelength=2, bbox_to_anchor=(0.32, 1.07), ncol=1, fontsize=FONT_DICT_SMALL['fontsize'], frameon=False)
    elif moment_name == 'KFM':
        ylim = format_kfm_y_ticks()
    for i_trial, trial_name in enumerate(['fpa', 'step_width', 'trunk_sway']):
        index = [i_trial*6, i_trial*6 + 2, i_trial*6 + 4]
        draw_sigifi_sign([mean_[x] for x in index], [std_[x] for x in index], [bar_locs[x] for x in index],
                         p_between_pattern[trial_name], ylim)


def finalize_f9(fig):
    plt.tight_layout(rect=[-0.01, -0.01, 1.06, 1.01], w_pad=2, h_pad=1)
    l1 = lines.Line2D([0.372, 0.372], [0.01, 0.85], linestyle='--', transform=fig.transFigure, color='gray')
    l2 = lines.Line2D([0.682, 0.682], [0.01, 0.85], linestyle='--', transform=fig.transFigure, color='gray')
    fig.lines.extend([l1, l2])
    save_fig('c5_gait_patterns')


if __name__ == "__main__":
    data_path = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\'
    result_date = 'results/0326'
    with h5py.File(result_date + 'KAM/8IMU_2camera/results.h5', 'r') as hf:
        kam_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kam_data_fields = json.loads(hf.attrs['columns'])
    with h5py.File(result_date + 'KFM/8IMU_2camera/results.h5', 'r') as hf:
        kfm_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kfm_data_fields = json.loads(hf.attrs['columns'])
    with h5py.File(data_path + 'gait_parameters.h5', 'r') as hf:
        gait_param_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        gait_param_fields = json.loads(hf.attrs['columns'])

    # get baseline parameter
    bl_param = np.zeros([len(SUBJECTS), len(gait_param_fields)])
    for i_subject, subject in enumerate(SUBJECTS):
        sub_param = gait_param_all_sub[subject]
        sub_bl_data = sub_param[sub_param[:, gait_param_fields.index('trial_id')] == 0, :]
        bl_param[i_subject] = np.mean(sub_bl_data, axis=0)
    bl_param_df = pd.DataFrame(bl_param, columns=gait_param_fields)

    fpa_config = {'trial_name': 'fpa', 'pattern_names': ('toe_in', 'normal_fpa', 'toe_out')}
    ts_config = {'trial_name': 'trunk_sway', 'pattern_names': ('small_ts', 'normal_ts', 'large_ts')}
    sw_config = {'trial_name': 'step_width', 'pattern_names': ('small_sw', 'normal_sw', 'large_sw')}
    fig, gs = init_f9()
    peak_of_gait_df = pd.DataFrame()
    # compute results
    for config in [fpa_config, ts_config, sw_config]:
        for data, data_fields, sign, moment_name in zip([kam_data_all_sub, kfm_data_all_sub], [kam_data_fields, kfm_data_fields], [1, -1], ['KAM', 'KFM']):
            for i_subject, subject in enumerate(SUBJECTS):
                sub_param, sub_data = gait_param_all_sub[subject], data[subject]
                sub_trial_data_sorted, param_sorted = sort_gait_cycles_according_to_param(config, sub_param, sub_data)
                small_steps, normal_steps, large_steps = np.array_split(sub_trial_data_sorted, 3)
                small_param, normal_param, large_param = np.array_split(param_sorted, 3)
                for steps, param, pattern_name in zip((small_steps, normal_steps, large_steps), (small_param, normal_param, large_param), config['pattern_names']):
                    # get_mean_gait_cycle_then_find_peak get_peak_of_each_gait_cycle
                    true_peak, pred_peak = get_peak_of_each_gait_cycle(np.stack(steps) * sign, data_fields, 0.5)
                    peak_of_gait_df = peak_of_gait_df.append(
                        [[i_subject, config['trial_name'], pattern_name, moment_name, np.mean(param),
                          bl_param_df[config['trial_name']][i_subject], np.mean(true_peak), np.mean(pred_peak)]])
    peak_of_gait_df.columns = ['subject_id', 'trial_name', 'pattern', 'moment', 'param_mean', 'param_bl', 'true_peak', 'pred_peak']

    # plot results
    for i_moment, moment_name in enumerate(['KAM', 'KFM']):
        moment_average, moment_std = [], []
        print(config)
        p_between_pattern = {config['trial_name']: [False, False, False] for config in [fpa_config, ts_config, sw_config]}
        for config in [fpa_config, sw_config, ts_config]:
            config_df = peak_of_gait_df[(peak_of_gait_df['moment'] == moment_name) & (peak_of_gait_df['trial_name'] == config['trial_name'])]
            for pattern_name in config['pattern_names']:
                pattern_df = config_df[config_df['pattern'] == pattern_name]
                moment_average.extend([pattern_df['true_peak'].mean(), pattern_df['pred_peak'].mean()])
                moment_std.extend([pattern_df['true_peak'].sem(), pattern_df['pred_peak'].sem()])
                p_between_models = round(ttest_rel(pattern_df['true_peak'], pattern_df['pred_peak']).pvalue, 3)
                print(p_between_models)
            for i_pair, pattern_pair in enumerate([(0, 1), (1, 2), (0, 2)]):
                p_val = ttest_rel(
                    config_df[config_df['pattern'] == config['pattern_names'][pattern_pair[0]]]['true_peak'].values,
                    config_df[config_df['pattern'] == config['pattern_names'][pattern_pair[1]]]['true_peak'].values).pvalue
                print('P between {} and {} is {}'.format(config['pattern_names'][pattern_pair[0]], config['pattern_names'][pattern_pair[1]], round(p_val, 3)))
                if p_val < 0.05:
                    p_between_pattern[config['trial_name']][i_pair] = True
        ax = fig.add_subplot(gs[i_moment, 0])
        draw_f9_subplot(moment_average, moment_std, p_between_pattern, ax, moment_name)
    finalize_f9(fig)
    plt.show()
