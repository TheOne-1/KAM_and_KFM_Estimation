import os
import h5py
import json
from PaperFigures import get_mean_std, format_plot
from const import SUBJECTS
import numpy as np
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc


def print_diff(result_dir, true_name, pred_name):

    with h5py.File(os.path.join(result_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])

    true_loc = _data_fields.index('true_' + true_name)
    true_data = np.concatenate([_data_all_sub[sub][:, :, true_loc] for sub in SUBJECTS])
    pred_loc = _data_fields.index('pred_' + pred_name)
    pred_data = np.concatenate([_data_all_sub[sub][:, :, pred_loc] for sub in SUBJECTS])

    diff = np.mean(np.abs(true_data - pred_data))
    print(diff, end='\t')


if __name__ == "__main__":
    result_dir1 = '/home/tan/VideoIMUCombined/experiment_data/KAM/training_results/2021-01-06 15:38:57.213976_14.2%'
    result_dir2 = '/home/tan/VideoIMUCombined/experiment_data/KAM/training_results/2021-01-07 01:21:17.840252'

    print('force_x_pre')
    print_diff(result_dir1, 'midout_force_x', 'midout_force_x_pre')
    print_diff(result_dir2, 'midout_force_x', 'midout_force_x_pre')
    print('\nforce_x')
    print_diff(result_dir1, 'midout_force_x', 'midout_force_x')
    print_diff(result_dir2, 'midout_force_x', 'midout_force_x')

    print('\nforce_z_pre')
    print_diff(result_dir1, 'midout_force_z', 'midout_force_z_pre')
    print_diff(result_dir2, 'midout_force_z', 'midout_force_z_pre')
    print('\nforce_z')
    print_diff(result_dir1, 'midout_force_z', 'midout_force_z')
    print_diff(result_dir2, 'midout_force_z', 'midout_force_z')

    print('\nr_x_pre')
    print_diff(result_dir1, 'midout_r_x', 'midout_r_x_pre')
    print_diff(result_dir2, 'midout_r_x', 'midout_r_x_pre')
    print('\nr_x')
    print_diff(result_dir1, 'midout_r_x', 'midout_r_x')
    print_diff(result_dir2, 'midout_r_x', 'midout_r_x')

    print('\nr_z_pre')
    print_diff(result_dir1, 'midout_r_z', 'midout_r_z_pre')
    print_diff(result_dir2, 'midout_r_z', 'midout_r_z_pre')
    print('\nr_z')
    print_diff(result_dir1, 'midout_r_z', 'midout_r_z')
    print_diff(result_dir2, 'midout_r_z', 'midout_r_z')

    print('\nmain_output')
    print_diff(result_dir1, 'main_output', 'main_output')
    print_diff(result_dir2, 'main_output', 'main_output')
