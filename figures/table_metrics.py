import os
import h5py
import json
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
from const import TRIALS, SUBJECTS
import matplotlib.pyplot as plt


def get_score(arr_true, arr_pred, w):
    assert(len(arr_true.shape) == 1 and arr_true.shape == arr_pred.shape == w.shape)

    locs = np.where(w.ravel())[0]
    arr_true, arr_pred = arr_true.ravel()[locs], arr_pred.ravel()[locs]
    plt.figure()
    plt.plot(arr_true, arr_pred, '.')
    plt.show()
    r2 = r2_score(arr_true, arr_pred)
    rmse = np.sqrt(mse(arr_true, arr_pred))
    r_rmse = rmse / (arr_true.max() - arr_true.min()) * 100
    cor_value = pearsonr(arr_true, arr_pred)[0]
    rmse = rmse / 9.81 * 100
    return {'R2': r2, 'RMSE percent': rmse, 'nRMSE percent': r_rmse, 'r':  cor_value}


if __name__ == '__main__':
    h5_dir = 'results/0107_KAM/IMU+OP'
    with h5py.File(os.path.join(h5_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])

    all_data = np.concatenate(list(_data_all_sub.values()), axis=0)
    all_data = pd.DataFrame(data=all_data.reshape([-1, all_data.shape[2]]), columns=_data_fields)
    mean_results, std_results = [], []
    for trial_index, trial_name in enumerate(TRIALS):
        trial_name = trial_name.replace('_', ' ')
        all_subject_results = []
        all_subject_results_tmp = []
        for subject_index, subject_name in enumerate(SUBJECTS):
            trial_loc = (all_data['trial_id'] == trial_index) & (all_data['subject_id'] == subject_index)
            true_value = all_data['true_main_output'][trial_loc]
            pred_value = all_data['pred_main_output'][trial_loc]
            weight = all_data['force_phase'][trial_loc]
            subject_result = get_score(true_value, pred_value, weight)
            all_subject_results.append(subject_result)
            all_subject_results_tmp.append({'subject': subject_name, 'category': trial_name, **subject_result})
        all_subject_results = {metric: [subject_result[metric] for subject_result in all_subject_results] for metric in all_subject_results[0].keys()}
        mean_result = {key:  '%.3f' % np.mean(value) for key, value in all_subject_results.items()}
        mean_results.append({'subject': 'all', 'category': trial_name, **mean_result})
        mean_results += all_subject_results_tmp
        std_result = {key:  '%.3f' % np.std(value) for key, value in all_subject_results.items()}
        std_results.append({'category': trial_name, **std_result})

    for export_file, results in [['KAM_mean_metrics.csv', mean_results], ['KAM_std_metrics.csv', std_results]]:
        result_file = os.path.join('./', export_file)
        with open(result_file, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(results[0].keys())
            f_csv.writerows([result.values() for result in results])

