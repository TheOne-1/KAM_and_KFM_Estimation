import h5py
import json
import pandas as pd
import numpy as np
from const import LINE_WIDTH, GRAVITY, FONT_DICT_SMALL, SUBJECTS, TRIALS
from figures.f6 import save_fig
from figures.PaperFigures import get_peak_of_each_gait_cycle, format_axis, get_mean_gait_cycle_then_find_peak
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
from more_itertools import sort_together
from scipy.stats import ttest_rel
import matplotlib.gridspec as gridspec


def draw_sigifi_sign(mean_, std_, bar_locs, p_between_pattern, ylim):
    one_two, two_three, one_three = p_between_pattern
    if len([p[0] for p in p_between_pattern if p[0]]) == 3:
        y_top = sorted([a + b for a, b in zip(mean_, std_)])[-3]
    else:
        y_top = sorted([a + b for a, b in zip(mean_, std_)])[-1]
    print([a + b for a, b in zip(mean_, std_)])
    top_lines = [y_top + 0.12 * ylim, y_top + 0.25 * ylim, y_top + 0.38 * ylim]
    for i_pair, [pair, loc_0, loc_1] in enumerate(zip([one_two, two_three, one_three], [0, 1, 0], [1, 2, 2])):
        if not pair[0] and not pair[1]: continue
        top_line = top_lines.pop(0)
        if loc_0 == 0 and loc_1 == 2:
            coe_0, coe_1 = 0.53, 0.47
        else:
            coe_0, coe_1 = 0.56, 0.44
        if pair[0]:
            plt.plot([bar_locs[2*loc_0], bar_locs[2*loc_1]], [top_line, top_line], color=color_0, linewidth=LINE_WIDTH)
        if pair[1]:
            plt.plot([bar_locs[2*loc_0+1], bar_locs[2*loc_1+1]], [top_line-0.025*ylim, top_line-0.025*ylim],
                     color=color_1, linewidth=LINE_WIDTH)
        plt.text(bar_locs[2*loc_0]*coe_0 + bar_locs[2 * loc_1 + 1] * coe_1, top_line - 0.072 * ylim, '*', fontdict={'fontname': 'Times New Roman'}, size=30, zorder=20)
        rect = patches.Rectangle((bar_locs[2*loc_0]*coe_0 + bar_locs[2 * loc_1 + 1] * coe_1, top_line - 0.097 * ylim), 0.42, 0.15*ylim, linewidth=0, color='white', zorder=10)
        ax.add_patch(rect)


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
    fig = plt.figure(figsize=(11, 5))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[5, 5])        # , width_ratios=[8, 1, 8]
    return fig, gs


def draw_f9_subplot(mean_, std_, p_between_pattern, ax, moment_name):
    def format_kam_y_ticks():
        ax.set_ylabel('1st Peak KAM (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
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
        ax.set_xlabel('Foot Progression Angle', fontdict=FONT_DICT_SMALL, labelpad=5)
        ax.set_xlim(-1, 8)
        ax.set_xticks(np.arange(0.5, 7, 3))
        ax.set_xticklabels(['Toe-in', 'Normal', 'Toe-out'],
                           fontdict=FONT_DICT_SMALL, linespacing=0.95)

    format_axis()
    format_x_ticks()
    bar_locs = [x + y for x in range(0, 9, 3) for y in [0, 1]]
    colors = [color_0, color_1] * int(len(bar_locs) / 2)
    for i_condition in range(len(mean_)):
        bar_.append(plt.bar(bar_locs[i_condition], mean_[i_condition], color=colors[i_condition], width=1))
    ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_,
                                               capsize=0, ecolor='black', fmt='none', lolims=True,
                                               elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 20)
    if moment_name == 'KAM':
        ylim = format_kam_y_ticks()
    elif moment_name == 'KFM':
        ylim = format_kfm_y_ticks()
    for i_trial, trial_name in enumerate(['fpa']):
        draw_sigifi_sign(mean_[6*i_trial:6*i_trial+6], std_[6*i_trial:6*i_trial+6], bar_locs[6*i_trial:6*i_trial+6],
                         p_between_pattern[trial_name], ylim)


def finalize_f9(fig):
    # pass
    plt.tight_layout(rect=[0, -0.01, 1, 0.9], w_pad=4, h_pad=1)
    plt.legend(bar_[0:2], ['Ground-Truth Moment', 'Estimated Moment'], frameon=False,
               bbox_to_anchor=(0.2, 1.24), ncol=2, fontsize=FONT_DICT_SMALL['fontsize'])
    save_fig('defense')


if __name__ == "__main__":
    data_path = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\'
    result_date = 'results/0326'
    color_0, color_1 = np.array([255, 166, 0]) / 255, np.array([0, 103, 137]) / 255
    with h5py.File(result_date + 'KAM/8IMU/results.h5', 'r') as hf:
        kam_data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        kam_data_fields = json.loads(hf.attrs['columns'])
    with h5py.File(result_date + 'KFM/8IMU/results.h5', 'r') as hf:
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
    bar_ = []
    fig, gs = init_f9()
    peak_of_gait_df = pd.DataFrame()
    # compute results
    for config in [fpa_config]:
        for data, data_fields, sign, moment_name in zip([kam_data_all_sub, kfm_data_all_sub], [kam_data_fields, kfm_data_fields], [1, -1], ['KAM', 'KFM']):
            for i_subject, subject in enumerate(SUBJECTS):
                sub_param, sub_data = gait_param_all_sub[subject], data[subject]
                sub_trial_data_sorted, param_sorted = sort_gait_cycles_according_to_param(config, sub_param, sub_data)
                small_steps, normal_steps, large_steps = np.array_split(sub_trial_data_sorted, 3)
                small_param, normal_param, large_param = np.array_split(param_sorted, 3)
                for steps, param, pattern_name in zip((small_steps, normal_steps, large_steps), (small_param, normal_param, large_param), config['pattern_names']):
                    # get_mean_gait_cycle_then_find_peak get_peak_of_each_gait_cycle
                    true_peak, pred_peak = get_mean_gait_cycle_then_find_peak(np.stack(steps) * sign, data_fields, 0.5)
                    peak_of_gait_df = peak_of_gait_df.append(
                        [[i_subject, config['trial_name'], pattern_name, moment_name, np.mean(param),
                          bl_param_df[config['trial_name']][i_subject], np.mean(true_peak), np.mean(pred_peak)]])
    peak_of_gait_df.columns = ['subject_id', 'trial_name', 'pattern', 'moment', 'param_mean', 'param_bl', 'true_peak', 'pred_peak']

    # plot results
    for i_moment, moment_name in enumerate(['KAM', 'KFM']):
        moment_average, moment_std = [], []
        print(config)
        p_between_pattern = {config['trial_name']: [False, False, False] for config in [fpa_config, ts_config, sw_config]}
        for config in [fpa_config]:
            config_df = peak_of_gait_df[(peak_of_gait_df['moment'] == moment_name) & (peak_of_gait_df['trial_name'] == config['trial_name'])]
            for pattern_name in config['pattern_names']:
                pattern_df = config_df[config_df['pattern'] == pattern_name]
                moment_average.extend([pattern_df['true_peak'].mean(), pattern_df['pred_peak'].mean()])
                moment_std.extend([pattern_df['true_peak'].std(), pattern_df['pred_peak'].std()])
                p_between_models = round(ttest_rel(pattern_df['true_peak'], pattern_df['pred_peak']).pvalue, 3)
                print(p_between_models)
            for i_pair, pattern_pair in enumerate([(0, 1), (1, 2), (0, 2)]):
                pattern_0_results = config_df[config_df['pattern'] == config['pattern_names'][pattern_pair[0]]]
                pattern_1_results = config_df[config_df['pattern'] == config['pattern_names'][pattern_pair[1]]]
                p_true = ttest_rel(pattern_0_results['true_peak'].values, pattern_1_results['true_peak'].values).pvalue
                p_pred = ttest_rel(pattern_0_results['pred_peak'].values, pattern_1_results['pred_peak'].values).pvalue
                print('P between {} and {} is {} (ground-truth) and {} (prediction)'.format(
                    config['pattern_names'][pattern_pair[0]], config['pattern_names'][pattern_pair[1]], round(p_true, 3), round(p_pred, 3)))
                p_between_pattern[config['trial_name']][i_pair] = p_true < 0.05, p_pred < 0.05
        ax = fig.add_subplot(gs[0, i_moment])
        draw_f9_subplot(moment_average, moment_std, p_between_pattern, ax, moment_name)
    finalize_f9(fig)
    plt.show()
