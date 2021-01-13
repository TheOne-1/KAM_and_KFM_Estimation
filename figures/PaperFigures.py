import numpy as np
import matplotlib.pyplot as plt
from const import LINE_WIDTH
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE, SUBJECTS
from base_kam_model import BaseModel
import h5py
import json


def format_plot(line_width=LINE_WIDTH):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=line_width)
    ax.yaxis.set_tick_params(width=line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def get_mean_std(data_array, data_fields, col_name):
    true_index, pred_index = data_fields.index('true_' + col_name), data_fields.index('pred_' + col_name)
    weight_index = data_fields.index(FORCE_PHASE)

    true_stance, _ = BaseModel.keep_stance_then_resample(data_array[:, :, true_index:true_index+1], data_array[:, :, weight_index:weight_index+1], 101)
    pred_stance, _ = BaseModel.keep_stance_then_resample(data_array[:, :, pred_index:pred_index+1], data_array[:, :, weight_index:weight_index+1], 101)
    true_mean, true_std = true_stance[:, :, 0].mean(axis=0), true_stance[:, :, 0].std(axis=0)
    pred_mean, pred_std = pred_stance[:, :, 0].mean(axis=0), pred_stance[:, :, 0].std(axis=0)
    return {'true_mean': true_mean, 'true_std': true_std, 'pred_mean': pred_mean, 'pred_std': pred_std}


def get_trial_data(file_path, trial_index, subjects=SUBJECTS):
    with h5py.File(file_path, 'r') as hf:
        _data_all = {subject: subject_data[:] for subject, subject_data in hf.items() if subject in subjects}
        _data_fields = json.loads(hf.attrs['columns'])

    trial_id_col_loc = _data_fields.index('trial_id')
    trial_data = [data[data[:, 0, trial_id_col_loc] == trial_index, :, :] for data in _data_all.values()]
    return trial_data, _data_fields

