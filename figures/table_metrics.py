import os
import h5py
import json
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS, TESTED_MODELS
from figures.PaperFigures import get_score, translate_moment_name


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


def get_all_results(test_folder_dir, moment_name):
    moment_name = translate_moment_name(moment_name)
    test_condition = test_folder_dir.split('/')[-1]
    with h5py.File(os.path.join(test_folder_dir, 'results.h5'), 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])
    all_data = np.concatenate(list(_data_all_sub.values()), axis=0)
    all_data = pd.DataFrame(data=all_data.reshape([-1, all_data.shape[2]]), columns=_data_fields)
    # all_data = all_data[(all_data.T != 0.).any()]
    all_results = []
    for trial_index, trial_name in enumerate(TRIALS):
        for subject_index, subject_name in enumerate(SUBJECTS):
            trial_loc = (all_data['trial_id'] == trial_index) & (all_data['subject_id'] == subject_index)
            true_value = all_data[moment_name][trial_loc]
            pred_value = all_data['pred_' + moment_name][trial_loc]
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
    result_date = 'results/1115/'
    kam_result = {model: get_all_results(result_date + model, 'KAM') for model in TESTED_MODELS}
    kfm_result = {model: get_all_results(result_date + model, 'KFM') for model in TESTED_MODELS}
    result_df_all_models = pd.DataFrame(kam_result[TESTED_MODELS[0]])[['subject', 'trial']]
    for result_dict, moment in zip([kam_result, kfm_result], ['KAM', 'KFM']):
        for result_name, results in result_dict.items():
            result_df = results[['MAE', 'RMSE', 'rRMSE', 'r']]
            result_df.columns = [column_name + '_' + result_name + '_' + moment for column_name in result_df.columns]
            result_df_all_models = pd.concat([result_df_all_models, result_df], axis=1)
    result_df_all_models.to_csv(result_date + '/estimation_result_individual.csv', index=False)
