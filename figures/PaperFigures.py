import numpy as np
import matplotlib.pyplot as plt
from const import LINE_WIDTH
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE, SUBJECTS
from base_kam_model import BaseModel
import h5py
import json
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr


def format_axis(line_width=LINE_WIDTH):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=line_width)
    ax.yaxis.set_tick_params(width=line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def hide_axis_add_grid():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color='lightgray', linewidth=1.5)
    ax.tick_params(color='lightgray', width=1.5)


def get_mean_std(data_array, data_fields, col_name):
    true_index, pred_index = data_fields.index('true_' + col_name), data_fields.index('pred_' + col_name)
    weight_index = data_fields.index(FORCE_PHASE)

    true_stance, _ = BaseModel.keep_stance_then_resample(data_array[:, :, true_index:true_index+1], data_array[:, :, weight_index:weight_index+1], 101)
    pred_stance, _ = BaseModel.keep_stance_then_resample(data_array[:, :, pred_index:pred_index+1], data_array[:, :, weight_index:weight_index+1], 101)
    true_mean, true_std = true_stance[:, :, 0].mean(axis=0), true_stance[:, :, 0].std(axis=0)
    pred_mean, pred_std = pred_stance[:, :, 0].mean(axis=0), pred_stance[:, :, 0].std(axis=0)
    return {'true_mean': true_mean, 'true_std': true_std, 'pred_mean': pred_mean, 'pred_std': pred_std}


def get_data(file_path, specific_trial=None, subjects=SUBJECTS):
    with h5py.File(file_path, 'r') as hf:
        _data_all = {subject: subject_data[:] for subject, subject_data in hf.items() if subject in subjects}
        _data_fields = json.loads(hf.attrs['columns'])

    if specific_trial is not None:
        trial_id_col_loc = _data_fields.index('trial_id')
        _data_all = [data[data[:, 0, trial_id_col_loc] == specific_trial, :, :] for data in _data_all.values()]
    return _data_all, _data_fields


def get_fpa(_data, _data_fields):
    def get_FPA_all():
        toe_col_loc = _data_fields.index('FM2_x')
        toe = _data[:, 0, toe_col_loc:toe_col_loc+3]
        x=1
    get_FPA_all()
        # forward_vector = toe - heel
        # if side == 'l':
        #     FPAs = - 180 / np.pi * np.arctan2(forward_vector[:, 0], forward_vector[:, 1])
        # else:
        #     FPAs = 180 / np.pi * np.arctan2(forward_vector[:, 0], forward_vector[:, 1])
        # return FPAs


def get_score(arr_true, arr_pred, w):
    assert(len(arr_true.shape) == 1 and arr_true.shape == arr_pred.shape == w.shape)
    locs = np.where(w.ravel())[0]
    arr_true, arr_pred = arr_true.ravel()[locs], arr_pred.ravel()[locs]
    mae = np.mean(np.abs(arr_true - arr_pred)) / 9.81 * 1000
    rmse = np.sqrt(mse(arr_true, arr_pred))
    r_rmse = rmse / (arr_true.max() - arr_true.min()) * 100
    cor_value = pearsonr(arr_true, arr_pred)[0]
    rmse = rmse / 9.81 * 1000  # TODO: error-prone code here. Modify generate_combined_data to calculate external KAM.
    return {'MAE': mae, 'RMSE': rmse, 'rRMSE': r_rmse, 'r':  cor_value}
