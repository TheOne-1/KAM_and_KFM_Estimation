from const import DATA_PATH, SUBJECTS, TRIALS, VICON_SAMPLE_RATE, SEGMENT_DEFINITIONS, SENSOR_LIST, IMU_FIELDS, \
    SAMPLE_INDEX, TRIAL_ID
import os
import pandas as pd
import numpy as np
import torch
import h5py
import json
from alan_framework import TfnNet, InertialNet, OutNet, VideoNet, LmfImuOnlyNet, ACC_ALL, GYR_ALL, VID_ALL, BaseFramework
from const import STATIC_DATA, HIGH_LEVEL_FEATURE, FORCE_PHASE, USED_KEYPOINTS, SUBJECT_HEIGHT, VIDEO_LIST
from figures.table_metrics import get_score, append_mean_results_in_the_end, get_metric_mean_std_result
from a_load_model_and_predict import normalize_array_separately
from placement_error_investigation import make_vid_relative_to_midhip, normalize_vid_by_size_of_subject_in_static_trial
from scipy.stats import sem


def print_relative_increase():
    results_df = pd.read_csv('sensor_failure_results.csv')
    metric_name = 'RMSE_'
    baseline_df = results_df[(results_df['trial'] == 'all') & (results_df['failure_node'] == 'no')]
    for sensor in ['CHEST', 'WAIST', 'L_THIGH', 'R_THIGH', 'L_SHANK', 'R_SHANK', 'L_FOOT', 'R_FOOT', '_90', '_180']:
        sensor_df = results_df[(results_df['trial'] == 'all') & (results_df['failure_node'] == sensor)]
        rmse_increase_kam = sensor_df[metric_name + 'KAM'].values - baseline_df[metric_name + 'KAM'].values
        rmse_increase_kfm = sensor_df[metric_name + 'KFM'].values - baseline_df[metric_name + 'KFM'].values
        print('{:.2f} ({:.2f})\t{:.2f} ({:.2f})'.format(
            rmse_increase_kam.mean(), sem(rmse_increase_kfm), rmse_increase_kfm.mean(), sem(rmse_increase_kfm)))


def print_absolute_values():
    results_df = pd.read_csv('sensor_failure_results.csv')
    metric_name = 'RMSE_'
    for sensor in ['CHEST', 'WAIST', 'L_THIGH', 'R_THIGH', 'L_SHANK', 'R_SHANK', 'L_FOOT', 'R_FOOT', '_90', '_180']:
        sensor_df = results_df[(results_df['trial'] == 'all') & (results_df['failure_node'] == sensor)]
        rmse_kam = sensor_df[metric_name + 'KAM'].values
        rmse_kfm = sensor_df[metric_name + 'KFM'].values
        print('{:.2f} ({:.2f})\t{:.2f} ({:.2f})'.format(
            rmse_kam.mean(), sem(rmse_kfm), rmse_kfm.mean(), sem(rmse_kfm)))


def replace_data_and_pred(data_fields, loc_set_zero, subject_data, model, model_inputs):
    vid_fields, i_high_level_start = VID_ALL, 0
    subject_data = np.array(subject_data, copy=True)
    subject_data[:, :, loc_set_zero] = 0
    antro_data = subject_data[:, :, [data_fields.index(field) for field in STATIC_DATA]]
    high_level_data = subject_data[:, :, [data_fields.index(field) for field in HIGH_LEVEL_FEATURE]]
    high_level_data = normalize_array_separately(high_level_data, model.scalars['high_level'], 'transform')
    model_inputs['others'] = torch.from_numpy(np.concatenate([antro_data, high_level_data[:, :, i_high_level_start:]], axis=2)).float().cuda()

    for name, fields in zip(['input_acc', 'input_gyr', 'input_vid'], [ACC_ALL, GYR_ALL, vid_fields]):
        data = subject_data[:, :, [data_fields.index(field) for field in fields]]
        data = normalize_array_separately(data, model.scalars[name], 'transform')
        model_inputs[name] = torch.from_numpy(data).float().cuda()
    predicted = model(model_inputs['input_acc'], model_inputs['input_gyr'], model_inputs['input_vid'],
                      model_inputs['others'], model_inputs['step_length']).cpu().detach().numpy()
    return predicted


def get_all_results(subject_data, data_fields, ground_truth_moment, predicted, weights, param_to_log):
    trials = []
    for i_trial, trial_name in enumerate(TRIALS):
        trial_series = pd.Series(param_to_log)
        trial_series['trial'] = trial_name
        for i_moment, moment_name in enumerate(['_KFM', '_KAM']):
            trial_id_loc = data_fields.index('trial_id')
            trial_loc = subject_data[:, 0, trial_id_loc] == i_trial
            true_value = ground_truth_moment[trial_loc, :, i_moment].ravel()
            pred_value = predicted[trial_loc, :, i_moment].ravel()
            weight = weights[trial_loc, :, i_moment].ravel()
            results = get_score(true_value, pred_value, weight)
            results = {key + moment_name: results[key] for key in results.keys()}
            trial_series = trial_series.append(pd.Series(results))
        trials.append(trial_series)

    mean_series = pd.Series(param_to_log)
    mean_series['trial'] = 'all'
    trials.append(mean_series.append(pd.DataFrame(trials).mean()))
    return trials


def replace_data_and_test():
    with h5py.File(DATA_PATH + '/40samples+stance.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    all_trials = []
    for model_name in model_names:
        for subject in SUBJECTS:
            print('replace_data_and_test, ' + model_name + ', ' + subject)
            model_path = os.path.join('..', 'figures', 'results', '1115', model_name, 'sub_models', subject, 'model.pth')
            model = torch.load(model_path).cuda()

            subject_data = data_all_sub[subject]
            ground_truth_moment = subject_data[:, :, [data_fields.index(field) for field in ['EXT_KM_X', 'EXT_KM_Y']]]
            weights = subject_data[:, :, [data_fields.index(FORCE_PHASE), data_fields.index(FORCE_PHASE)]]
            model_inputs = {'step_length': torch.from_numpy(np.sum(~(subject_data[:, :, 0] == 0.), axis=1))}
            subject_data = make_vid_relative_to_midhip(subject_data, data_fields)
            subject_data = normalize_vid_by_size_of_subject_in_static_trial(subject_data, data_fields)

            """ No data zeroing """
            predicted = replace_data_and_pred(data_fields, [], subject_data, model, model_inputs)
            param_to_log = {'subject': subject, 'model_name': model_name, 'failure_node': 'no'}
            all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted, weights, param_to_log))

            """ Setting single IMU to zero  """
            for sensor in SENSOR_LIST:
                imu_col_loc = [data_fields.index(axis + '_' + sensor) for axis in IMU_FIELDS[:6]]
                predicted = replace_data_and_pred(data_fields, imu_col_loc, subject_data, model, model_inputs)
                param_to_log = {'subject': subject, 'model_name': model_name, 'failure_node': sensor}
                all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted,
                                                  weights, param_to_log))

            """ Setting single IMU to zero  """
            for camera in ['_90', '_180']:
                camera_col_loc = [data_fields.index(marker + axis + camera) for axis in ['_x', '_y'] for marker in VIDEO_LIST]
                predicted = replace_data_and_pred(data_fields, camera_col_loc, subject_data, model, model_inputs)
                param_to_log = {'subject': subject, 'model_name': model_name, 'failure_node': camera}
                all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted,
                                                  weights, param_to_log))


    results_df = pd.DataFrame(all_trials)
    results_df.to_csv('sensor_failure_results.csv', index=False)


model_names = ['TfnNet']
if __name__ == "__main__":
    replace_data_and_test()
    print_absolute_values()

