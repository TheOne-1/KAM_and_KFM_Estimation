import h5py
import json
import pandas as pd
import numpy as np
from const import LINE_WIDTH, FONT_DICT_SMALL, SUBJECTS, TRIALS
from figures.PaperFigures import save_fig
from figures.PaperFigures import get_peak_of_each_gait_cycle, format_axis, get_mean_gait_cycle_then_find_peak
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as lines
from more_itertools import sort_together
from scipy.stats import ttest_rel
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def sort_gait_cycles_according_to_param(param_config, sub_param, sub_data):
    ts_id = TRIALS.index(param_config['trial_name'])
    sub_ts_all = sub_param[sub_param[:, gait_param_fields.index('trial_id')] == ts_id, gait_param_fields.index(param_config['trial_name'])]
    sub_trial_data = sub_data[sub_data[:, 0, data_fields.index('trial_id')] == ts_id, :, :]
    param_sorted, sub_trial_data_sorted = sort_together((sub_ts_all, sub_trial_data))
    return sub_trial_data_sorted, param_sorted


def init_f10():
    rc('font', family='Arial')
    fig = plt.figure(figsize=(16, 5.9))
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[5, 5])        # , width_ratios=[8, 1, 8]
    return fig, gs


def draw_f10_subplot(true_peak_, pred_peak_, subjects_, ax, moment_name_):
    def format_kam_ticks():
        ax.set_xlabel('$\Delta$ Peak KAM: Ground-Truth (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('$\Delta$ Peak KAM: Proposed Model (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ylim_lo, ylim_hi = -3, 4
        ax.set_xlim(ylim_lo, ylim_hi)
        ax.set_xticks(range(ylim_lo, ylim_hi+1))
        ax.set_xticklabels(range(ylim_lo, ylim_hi+1), fontdict=FONT_DICT_SMALL)
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.set_yticks(range(ylim_lo, ylim_hi+1))
        ax.set_yticklabels(range(ylim_lo, ylim_hi+1), fontdict=FONT_DICT_SMALL)

    def format_kfm_ticks():
        ax.set_xlabel('$\Delta$ Peak KFM: Ground-Truth (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('$\Delta$ Peak KFM: Proposed Model (%BW$\cdot$BH)', fontdict=FONT_DICT_SMALL)
        ylim_lo, ylim_hi = -5, 9
        ax.set_xlim(ylim_lo, ylim_hi)
        ax.set_xticks(range(ylim_lo, ylim_hi+1, 2))
        ax.set_xticklabels(range(ylim_lo, ylim_hi+1, 2), fontdict=FONT_DICT_SMALL)
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.set_yticks(range(ylim_lo, ylim_hi+1, 2))
        ax.set_yticklabels(range(ylim_lo, ylim_hi+1, 2), fontdict=FONT_DICT_SMALL)

    rc('font', family='Arial')
    sub_colors = ['C1', 'C2', 'C0']
    format_axis()
    dot_plot, all_dots_true, all_dots_pred = [], [], []
    for i_sub, subject in enumerate(subjects_):
        dot_plot.append(plt.scatter(true_peak_[subject], pred_peak_[subject], c=sub_colors[i_sub], edgecolors='none',
                                    s=40, marker='.', alpha=1))
        all_dots_true.extend(true_peak_[subject])
        all_dots_pred.extend(pred_peak_[subject])
    if moment_name_ == 'KAM':
        format_kam_ticks()
    else:
        format_kfm_ticks()

    coef = np.polyfit(all_dots_true, all_dots_pred, 1)
    poly1d_fn = np.poly1d(coef)
    black_line, = plt.plot([-10, 10], poly1d_fn([-10, 10]), color='black', linewidth=LINE_WIDTH, alpha=0.5)
    # RMSE = np.sqrt(mse(np.array(all_dots_true), np.array(all_dots_pred)))
    # ax.text(0.6, 0.05, 'ρ = {:4.2f}\nRMSE = {:4.2f} (%BW·BH)'.format(correlation, RMSE),
    #         fontdict=FONT_DICT_SMALL, transform=ax.transAxes)
    # correlation = pearsonr(all_dots_true, all_dots_pred)[0]
    r2 = r2_score(all_dots_true, all_dots_pred)
    plt.text(0.578, 0.11, '$R^2$ = {:4.2f}'.format(r2), fontdict=FONT_DICT_SMALL, transform=ax.transAxes)
    plt.text(0.6, 0.04, '$y$ = {:4.2f}$x$ + {:4.2f}'.format(coef[0], coef[1]), fontdict=FONT_DICT_SMALL, transform=ax.transAxes)
    return dot_plot, black_line


def finalize_f10(dot_plot, black_line):
    plt.tight_layout(rect=[0., -0.02, 1., 0.9], w_pad=6)
    legend_0 = plt.legend(dot_plot, ['Subject {}'.format(i_sub+1) for i_sub in range(3)], handlelength=3, bbox_to_anchor=(-0.05, 1.19),
               frameon=False, markerscale=3, handletextpad=-0.5, fontsize=FONT_DICT_SMALL['fontsize'], ncol=3)
    plt.legend([black_line], ['Regression Line'], handlelength=3, bbox_to_anchor=(0.47, 1.19),
               frameon=False, fontsize=FONT_DICT_SMALL['fontsize'])
    plt.gca().add_artist(legend_0)
    save_fig('f10', 600)


if __name__ == "__main__":
    data_path = 'J:\Projects\VideoIMUCombined\experiment_data\KAM\\'
    color_0, color_1 = np.array([90, 140, 20]) / 255, np.array([0, 103, 137]) / 255       # [255, 166, 0]
    with h5py.File('results/1115/TfnNet/results.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])
    with h5py.File(data_path + 'gait_parameters.h5', 'r') as hf:
        gait_param_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        gait_param_fields = json.loads(hf.attrs['columns'])

    # get baseline parameter
    bl_param = np.zeros([len(SUBJECTS), len(gait_param_fields)])
    for i_subject, subject in enumerate(SUBJECTS):
        sub_param = gait_param_all_sub[subject]
        sub_bl_param = sub_param[sub_param[:, gait_param_fields.index('trial_id')] == 0, :]
        bl_param[i_subject] = np.mean(sub_bl_param, axis=0)
    bl_param_df = pd.DataFrame(bl_param, columns=gait_param_fields)

    fpa_config = {'trial_name': 'fpa', 'pattern_names': ('toe_in', 'normal_fpa', 'toe_out')}
    ts_config = {'trial_name': 'trunk_sway', 'pattern_names': ('small_ts', 'normal_ts', 'large_ts')}
    sw_config = {'trial_name': 'step_width', 'pattern_names': ('small_sw', 'normal_sw', 'large_sw')}
    fig, gs = init_f10()
    true_peak_subs, pred_peak_subs = {}, {}
    for i_moment, moment_name in enumerate(['KAM', 'KFM']):
        for i_subject, subject in enumerate(SUBJECTS):
            true_peak_sub, pred_peak_sub = [], []
            sub_param, sub_data = gait_param_all_sub[subject], data_all_sub[subject]
            sub_bl_data = sub_data[sub_data[:, 0, data_fields.index('trial_id')] == 0, :, :]
            true_peaks_bl, pred_peaks_bl = get_peak_of_each_gait_cycle(np.stack(sub_bl_data), data_fields, moment_name, 0.5)
            true_peaks_bl_mean, pred_peaks_bl_mean = np.mean(true_peaks_bl), np.mean(pred_peaks_bl)
            sub_three_trial_data = []
            for config in [fpa_config, ts_config, sw_config]:
                sub_trial_data_sorted, param_sorted = sort_gait_cycles_according_to_param(config, sub_param, sub_data)
                sub_three_trial_data.extend(sub_trial_data_sorted)
            true_peak, pred_peak = get_peak_of_each_gait_cycle(np.stack(sub_three_trial_data), data_fields, moment_name, 0.5)
            true_peak_sub.extend([element - true_peaks_bl_mean for element in true_peak])
            pred_peak_sub.extend([element - pred_peaks_bl_mean for element in pred_peak])
            true_peak_subs[subject], pred_peak_subs[subject] = true_peak_sub, pred_peak_sub
        folder_1_subs = ['s011_wuxingze', 's004_ouyangjue', 's009_sunyubo']
        ax = fig.add_subplot(gs[i_moment])
        dot_plot, black_line = draw_f10_subplot(true_peak_subs, pred_peak_subs, folder_1_subs, ax, moment_name)

        r2_each_sub = []
        true_dots_all, pred_dots_all = [], []
        for subject in SUBJECTS:
            r2_each_sub.append(r2_score(true_peak_subs[subject], pred_peak_subs[subject]))
            true_dots_all.extend(true_peak_subs[subject])
            pred_dots_all.extend(pred_peak_subs[subject])
        print(np.mean(r2_each_sub))
        print(r2_each_sub)
        r2_all = r2_score(true_dots_all, pred_dots_all)
        print(r2_all)

    finalize_f10(dot_plot, black_line)
    plt.show()
