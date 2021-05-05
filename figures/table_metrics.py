import os
import h5py
import json
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS, SENSOR_COMBINATION
from figures.PaperFigures import get_score


def append_mean_results_in_the_end(all_results):
    all_result_df = pd.DataFrame(all_results)
    for subject_index, subject_name in enumerate(SUBJECTS):
        sub_mean_df = pd.DataFrame(all_result_df[all_result_df['subject'] == subject_name].mean()).T
        sub_mean_df['subject'] = subject_name
        sub_mean_df['trial'] = 'all'
        all_result_df = all_result_df.append(sub_mean_df, ignore_index=True)
    return all_result_df


def get_metric_mean_std_result(all_result_df, metric):
    metric_df = all_result_df[(all_result_df['trial'] == 'all')][metric]
    return metric_df.mean(), metric_df.sem()


def get_all_results(test_folder_dir, test_condition):
    with h5py.File(os.path.join(test_folder_dir, test_condition, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])
    all_data = np.concatenate(list(_data_all_sub.values()), axis=0)
    all_data = pd.DataFrame(data=all_data.reshape([-1, all_data.shape[2]]), columns=_data_fields)
    # all_data = all_data[(all_data.T != 0.).any()]
    all_results = []
    for trial_index, trial_name in enumerate(TRIALS):
        # if trial_index == 0: continue
        for subject_index, subject_name in enumerate(SUBJECTS):
            trial_loc = (all_data['trial_id'] == trial_index) & (all_data['subject_id'] == subject_index)
            true_value = all_data['true_main_output'][trial_loc]
            pred_value = all_data['pred_main_output'][trial_loc]
            weight = all_data['force_phase'][trial_loc]
            subject_result = get_score(true_value, pred_value, weight)
            all_results.append({'subject': subject_name, 'trial': trial_name, **subject_result})
    all_result_df = append_mean_results_in_the_end(all_results)
    mean_std_RMSE = get_metric_mean_std_result(all_result_df, 'RMSE')
    mean_std_rRMSE = get_metric_mean_std_result(all_result_df, 'rRMSE')
    mean_std_r = get_metric_mean_std_result(all_result_df, 'r')
    print("Overall RMSE (%BWxBH), rRMSE, and correlation coefficient were " +
          "%.2f ± %.2f, " % mean_std_RMSE +
          "%.1f ± %.1f, " % mean_std_rRMSE +
          "and %.2f ± %.2f" % mean_std_r +
          " for {} condition".format(test_condition))
    return all_result_df


if __name__ == '__main__':
    result_date = 'results/0326'       # all_feature_
    for target in ['KAM', 'KFM']:
        print(target)
        combinations = SENSOR_COMBINATION   # ['8IMU_2camera', '8IMU', '3IMU_2camera', '1IMU_2camera', '2camera']
        combo_result = [get_all_results(result_date + target, sensor) for sensor in combinations]

        result_df_all_three = pd.DataFrame(combo_result[0])[['subject', 'trial']]
        for results, result_name in zip(combo_result, combinations):
            result_df = results[['MAE', 'RMSE', 'rRMSE', 'r']]
            result_df.columns = [column_name + '_' + result_name for column_name in result_df.columns]
            result_df_all_three = pd.concat([result_df_all_three, result_df], axis=1)
        result_df_all_three.to_csv(result_date + target + '/estimation_result_individual.csv', index=False)
