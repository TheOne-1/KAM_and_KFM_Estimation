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


def get_all_results(h5_dir):
    with h5py.File(os.path.join(h5_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])
    all_data = np.concatenate(list(_data_all_sub.values()), axis=0)
    all_data = pd.DataFrame(data=all_data.reshape([-1, all_data.shape[2]]), columns=_data_fields)
    all_results = []
    for trial_index, trial_name in enumerate(TRIALS + ['all']):
        for subject_index, subject_name in enumerate(SUBJECTS):
            if trial_name == 'all':
                trial_loc = all_data['subject_id'] == subject_index
            else:
                trial_loc = (all_data['trial_id'] == trial_index) & (all_data['subject_id'] == subject_index)
            true_value = all_data['true_main_output'][trial_loc]
            pred_value = all_data['pred_main_output'][trial_loc]
            weight = all_data['force_phase'][trial_loc]
            subject_result = get_score(true_value, pred_value, weight)
            all_results.append({'subject': subject_name, 'trial': trial_name, **subject_result})
    mean_std_r = get_overall_mean_std_result(all_results, 'r')
    mean_std_rRMSE = get_overall_mean_std_result(all_results, 'rRMSE')
    mean_std_RMSE = get_overall_mean_std_result(all_results, 'RMSE')
    mean_std_MAE = get_overall_mean_std_result(all_results, 'MAE')
    print("Correlation coefficient, rRMSE, RMSE(10^-3) and MAE(10^-3) for overall trials were " +
          "{}, ".format(mean_std_r) +
          "{}, ".format(mean_std_rRMSE) +
          "{}, ".format(mean_std_RMSE) +
          "{}, respectively.".format(mean_std_MAE))
    return all_results


if __name__ == '__main__':
    for target in ['KAM', 'KFM']:
        IMU_OP_results, IMU_results, OP_results = [get_all_results('results/0120_' + target + '/' + sensor)
                                                   for sensor in ['IMU+OP', 'IMU', 'OP']]

        get_trial_result = lambda all_results, trial_name: list(filter(lambda result: result['trial'] == trial_name, all_results))
        results = {}
        for _input, input_name in [[IMU_results, 'IMU'], [IMU_OP_results, 'Combined'], [OP_results, 'Camera']]:
            trial_results = {trial: get_overall_mean_std_result(get_trial_result(_input, trial), 'rRMSE') for trial in TRIALS}
            results[input_name] = trial_results

        for trial_index, trial in enumerate(TRIALS):
            print('& '+TRIALS_PRINT[trial_index], end=' ')
            for source in ['Combined', 'IMU', 'Camera']:
                print(' & ' + results[source][trial], end='')
            print('\\\\')
        result_df_all_three = pd.DataFrame(IMU_OP_results)[['subject', 'trial']]
        for results, result_name in zip([IMU_OP_results, IMU_results, OP_results], ['_IMU_OP', '_IMU', '_OP']):
            result_df = pd.DataFrame(results)[['MAE', 'RMSE', 'rRMSE', 'r']]
            result_df.columns = [column_name + result_name for column_name in result_df.columns]
            result_df_all_three = pd.concat([result_df_all_three, result_df], axis=1)
        result_df_all_three.to_csv('./exports/' + target + '_estimation_result_individual.csv', index=False)
