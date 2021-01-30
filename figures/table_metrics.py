import os
import h5py
import json
import csv
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS, SENSOR_COMBINATION
import matplotlib.pyplot as plt
from PaperFigures import get_score


def get_overall_mean_std_result(all_results, metric):
    return np.mean([result[metric] for result in all_results]), np.std([result[metric] for result in all_results])


def get_all_results(test_folder_dir, test_condition):
    with h5py.File(os.path.join(test_folder_dir, test_condition, 'results.h5'), 'r') as hf:
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
    mean_std_RMSE = get_overall_mean_std_result(all_results, 'RMSE')
    mean_std_rRMSE = get_overall_mean_std_result(all_results, 'rRMSE')
    mean_std_r = get_overall_mean_std_result(all_results, 'r')
    print("Overall RMSE (10^-3), rRMSE, and correlation coefficient were " +
          "%.1f ± %.1f, " % mean_std_RMSE +
          "%.1f ± %.1f, " % mean_std_rRMSE +
          "and %.2f ± %.2f" % mean_std_r +
          " for {} condition".format(test_condition))
    return all_results


if __name__ == '__main__':
    result_date = 'results/0127_selected_feature_'       # all_feature_
    for target in ['KAM', 'KFM']:
        combo_result = [get_all_results(result_date + target, sensor) for sensor in SENSOR_COMBINATION]

        get_trial_result = lambda all_results, trial_name: list(filter(lambda result: result['trial'] == trial_name, all_results))
        results = {}
        for _input, input_name in zip(combo_result, SENSOR_COMBINATION):
            trial_results = {trial: get_overall_mean_std_result(get_trial_result(_input, trial), 'rRMSE') for trial in TRIALS}
            results[input_name] = trial_results

        result_df_all_three = pd.DataFrame(combo_result[0])[['subject', 'trial']]
        for results, result_name in zip(combo_result, SENSOR_COMBINATION):
            result_df = pd.DataFrame(results)[['MAE', 'RMSE', 'rRMSE', 'r']]
            result_df.columns = [column_name + '_' + result_name for column_name in result_df.columns]
            result_df_all_three = pd.concat([result_df_all_three, result_df], axis=1)
        result_df_all_three.to_csv(result_date + target + '/estimation_result_individual.csv', index=False)
