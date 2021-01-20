import os
import h5py
import json
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from const import TRIALS, SUBJECTS, TRIALS_PRINT
import matplotlib.pyplot as plt


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


def get_overall_mean_std_result(all_results, metric):
    return '%.2f' % np.mean([result[metric] for result in all_results]) + ' (' + '%.2f' % np.std([result[metric] for result in all_results]) + ')'


def get_peak_results(all_data, data_fields):
    i = lambda s: data_fields.index(s)
    peak_results = np.zeros([all_data.shape[0], 4])
    for step_index in range(all_data.shape[0]):
        force_loc = all_data[step_index, :, i('force_phase')] == 1
        subject_id = all_data[step_index, 0, i('subject_id')]
        trial_id = all_data[step_index, 0, i('trial_id')]
        peak_kam_pred = all_data[step_index, force_loc, i('pred_main_output')].max()
        peak_kam_true = all_data[step_index, force_loc, i('true_main_output')].max()
        peak_results[step_index, :] = [subject_id, trial_id,  peak_kam_pred, peak_kam_true]
    return pd.DataFrame(peak_results, columns=['subject_id', 'trial_id', 'pred_main_output', 'true_main_output'])


def get_all_results(h5_dir):
    with h5py.File(os.path.join(h5_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])
    all_data = np.concatenate(list(_data_all_sub.values()), axis=0)
    peak_kams = get_peak_results(all_data, _data_fields)
    all_data = pd.DataFrame(data=all_data.reshape([-1, all_data.shape[2]]), columns=_data_fields)
    profile_results, peak_results = [], []
    for trial_index, trial_name in enumerate(TRIALS + ['all']):
        for subject_index, subject_name in enumerate(SUBJECTS):
            if trial_name == 'all':
                trial_loc = all_data['subject_id'] == subject_index
                p_trial_loc = peak_kams['subject_id'] == subject_index
            else:
                trial_loc = (all_data['trial_id'] == trial_index) & (all_data['subject_id'] == subject_index)
                p_trial_loc = (peak_kams['trial_id'] == trial_index) & (peak_kams['subject_id'] == subject_index)
            true_value = all_data['true_main_output'][trial_loc]
            pred_value = all_data['pred_main_output'][trial_loc]
            weight = all_data['force_phase'][trial_loc]
            profile_results.append({'subject': subject_name, 'trial': trial_name, **get_score(true_value, pred_value, weight)})

            p_true_value = peak_kams['true_main_output'][p_trial_loc]
            p_pred_value = peak_kams['pred_main_output'][p_trial_loc]
            p_weight = np.ones(p_true_value.shape)
            peak_results.append({'subject': subject_name, 'trial': trial_name, **get_score(p_true_value, p_pred_value, p_weight)})

    def print_trial_results(results):
        mean_std_r = get_overall_mean_std_result(results, 'r')
        mean_std_rRMSE = get_overall_mean_std_result(results, 'rRMSE')
        mean_std_RMSE = get_overall_mean_std_result(results, 'RMSE')
        mean_std_MAE = get_overall_mean_std_result(results, 'MAE')
        print("Correlation coefficient, rRMSE, RMSE(10^-3) and MAE(10^-3) for overall trials were " +
              "{}, ".format(mean_std_r) +
              "{}, ".format(mean_std_rRMSE) +
              "{}, ".format(mean_std_RMSE) +
              "{}, respectively.".format(mean_std_MAE))

    print('Profile KAM', end=' ')
    print_trial_results(profile_results)
    print('Peak KAM', end=' ')
    print_trial_results(peak_results)
    return profile_results


if __name__ == '__main__':
    for target in ['KAM', 'KFM']:
        IMU_OP_results, IMU_results, OP_results = [get_all_results('results/0107_' + target + '/' + sensor)
                                                   for sensor in ['IMU+OP', 'IMU', 'OP']]

        with open(os.path.join('./exports', target + '_estimation_result.csv'), 'w') as f:
            f_csv = csv.writer(f)
            get_trial_result = lambda all_results, trial_name: list(filter(lambda result: result['trial'] == trial_name, all_results))
            f_csv.writerow(['', *TRIALS_PRINT])
            for _input, input_name in [[IMU_results, 'IMU'], [IMU_OP_results, 'Combined'], [OP_results, 'Camera']]:
                trial_results = [get_overall_mean_std_result(get_trial_result(_input, trial), 'rRMSE') for trial in TRIALS]
                f_csv.writerow([input_name, *trial_results])

        result_df_all_three = pd.DataFrame(IMU_OP_results)[['subject', 'trial']]
        for results, result_name in zip([IMU_OP_results, IMU_results, OP_results], ['_IMU_OP', '_IMU', '_OP']):
            result_df = pd.DataFrame(results)[['MAE', 'RMSE', 'rRMSE', 'r']]
            result_df.columns = [column_name + result_name for column_name in result_df.columns]
            result_df_all_three = pd.concat([result_df_all_three, result_df], axis=1)
        result_df_all_three.to_csv('./exports/' + target + '_estimation_result_individual.csv', index=False)
