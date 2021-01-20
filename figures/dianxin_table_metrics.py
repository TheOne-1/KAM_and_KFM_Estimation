import os
import h5py
import json
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from const import TRIALS, SUBJECTS, TRIALS_PRINT


def get_score(arr_true, arr_pred, w):
    assert(len(arr_true.shape) == 1 and arr_true.shape == arr_pred.shape == w.shape)

    locs = np.where(w.ravel())[0]
    arr_true, arr_pred = arr_true.ravel()[locs], arr_pred.ravel()[locs]
    mae = np.mean(np.abs(arr_true - arr_pred)) / 9.81 * 1000
    rmse = np.sqrt(mse(arr_true, arr_pred))
    r_rmse = rmse / (arr_true.max() - arr_true.min()) * 100
    cor_value = pearsonr(arr_true, arr_pred)[0]
    rmse = rmse / 9.81 * 1000
    return {'RMSE': rmse, 'rRMSE': r_rmse, 'r':  cor_value, 'MAE': mae}


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
        for subject_name in _data_all_sub:
            subject_index = SUBJECTS.index(subject_name)
            if trial_name == 'all':
                trial_loc = all_data['subject_id'] == subject_index
            else:
                trial_loc = (all_data['trial_id'] == trial_index) & (all_data['subject_id'] == subject_index)
            true_value = all_data['true_main_output'][trial_loc]
            pred_value = all_data['pred_main_output'][trial_loc]
            weight = all_data['force_phase'][trial_loc]
            subject_result = get_score(true_value, pred_value, weight)
            all_results.append({'subject': subject_name, 'trial': trial_name, **subject_result})
    if 'KAM' in h5_dir:
        pd.DataFrame(all_results).to_csv('./exports/KAM_individual_result_' + h5_dir.split('/')[-1] + '.csv')
    else:
        pd.DataFrame(all_results).to_csv('./exports/KFM_individual_result_' + h5_dir.split('/')[-1] + '.csv')
    mean_std_r = get_overall_mean_std_result(all_results, 'r')
    mean_std_rRMSE = get_overall_mean_std_result(all_results, 'rRMSE')
    mean_std_RMSE = get_overall_mean_std_result(all_results, 'RMSE')
    mean_std_MAE = get_overall_mean_std_result(all_results,  'MAE')
    print("MAE(10^-3), RMSE(10^-3), rRMSE(%) and Correlation coefficient for overall trials were " +
          "{}, ".format(mean_std_MAE) +
          "{}, ".format(mean_std_RMSE) +
          "{}, ".format(mean_std_rRMSE) +
          "{}, respectively.".format(mean_std_r))
    return all_results


if __name__ == '__main__':
    KAM_results, KFM_results = [get_all_results('results/' + target + '_12sub/') for target in ['KAM', 'KFM']]
    with open(os.path.join('./exports/dianxin_estimation_result.csv'), 'w') as f:
        f_csv = csv.writer(f)
        for _input, input_name in [[KAM_results, 'KAM'], [KFM_results, 'KFM']]:
            get_trial_result = lambda all_results, trial_name: list(filter(lambda result: result['trial'] == trial_name, all_results))
            f_csv.writerow([input_name, 'MAE(10^-3)', 'RMSE(10^-3)', 'rRMSE(%)', 'r'])
            metrics = ['MAE', 'RMSE', 'rRMSE', 'r']
            for trial_index, trial in enumerate(TRIALS):
                trial_results = [get_overall_mean_std_result(get_trial_result(_input, trial), metric) for metric in metrics]
                f_csv.writerow([TRIALS_PRINT[trial_index], *trial_results])

