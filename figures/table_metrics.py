import os
import h5py
import json
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
from const import TRIALS
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
    r_rmse = rmse / (arr_true.max() - arr_true.min())
    cor_value = pearsonr(arr_true, arr_pred)[0]
    return {'R2': '%.3f' % r2, 'RMSE': '%.3f' % rmse, 'nRMSE': '%.3f' % r_rmse, 'r': '%.3f' % cor_value}


if __name__ == '__main__':
    h5_dir = './results/0107_KAM/IMU+OP'
    with h5py.File(os.path.join(h5_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])

    all_data = np.concatenate(list(_data_all_sub.values()), axis=0)
    all_data = pd.DataFrame(data=all_data.reshape([-1, all_data.shape[2]]), columns=_data_fields)
    results = []
    for trial_index, trial_name in enumerate(TRIALS):
        trial_name = trial_name.replace('_', ' ')
        trial_loc = all_data['trial_id'] == trial_index
        true_value = all_data['true_main_output'][trial_loc]
        pred_value = all_data['pred_main_output'][trial_loc]
        weight = all_data['force_phase'][trial_loc]
        scores = {'category': trial_name, **get_score(true_value, pred_value, weight)}
        results.append(scores)
    scores = {'category': 'all', **get_score(all_data['true_main_output'], all_data['pred_main_output'], all_data['force_phase'])}
    results.append(scores)
    result_file = os.path.join('./', 'KAM_metrics.csv')
    with open(result_file, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(results[0].keys())
        f_csv.writerows([result.values() for result in results])
