import numpy as np
import matplotlib.pyplot as plt
from const import LINE_WIDTH, FONT_DICT_SMALL, FONT_SIZE, FONT_DICT, FONT_DICT_X_SMALL, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE
from base_kam_model import BaseModel


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
    return np.array([true_mean, true_std, pred_mean, pred_std])


