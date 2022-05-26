from const import DATA_PATH, SUBJECTS, TRIALS, SENSOR_LIST, IMU_FIELDS, \
    SAMPLE_INDEX, TRIAL_ID, STATIC_DATA, HIGH_LEVEL_FEATURE, FORCE_PHASE, SEGMENT_DEFINITIONS, VIDEO_LIST
import os
import pandas as pd
import numpy as np
import torch
import h5py
import json
from alan_framework import TfnNet, InertialNet, OutNet, VideoNet, LmfImuOnlyNet, ACC_ALL, GYR_ALL, VID_ALL, BaseFramework
from figures.table_metrics import get_score
from a_load_model_and_predict import normalize_array_separately
from placement_error_investigation import Simulator, R_i0_g, make_vid_relative_to_midhip
from placement_error_investigation import normalize_vid_by_size_of_subject_in_static_trial
import matplotlib.gridspec as gridspec
from figures.PaperFigures import format_axis
from const import LINE_WIDTH,FONT_DICT_SMALL
import matplotlib.pyplot as plt
from matplotlib import rc
from transforms3d.axangles import axangle2mat
import random


params_90 = {'init_rot': np.array([[0., 1, 0], [0, 0, -1], [-1, 0, 0]]),
             'init_tvec': np.array([-900., 1200, 3600])}
params_180 = {'init_rot': np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]),
              'init_tvec': np.array([-600., 1200, 2100])}


class ImuPlacementSimulater(Simulator):
    def get_delta_acc_and_R(self, imu_loc_during_static, x_mm=0., y_mm=0., z_mm=0.):
        R_relative_to_static, imu_trajectory = self._simulate_virtual_marker_trajectory(imu_loc_during_static)
        _, imu_trajectory_with_position_error = self._simulate_virtual_marker_trajectory(imu_loc_during_static + np.array([x_mm, y_mm, z_mm], dtype=np.float32))
        delta_acc = self._get_delta_acc_in_earth_frame(imu_trajectory, imu_trajectory_with_position_error)
        return delta_acc, R_relative_to_static
    
    def add_delta_acc_to_data(self, delta_acc, R_relative_to_static, R_i0_g=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        simulated_data = np.array(self.imu_data, copy=True)
        data_len = R_relative_to_static.shape[0]
        for i_sample in range(data_len):
            R_i_g = np.matmul(R_relative_to_static[i_sample], R_i0_g)
            acc_ori = simulated_data[i_sample, :3]
            simulated_data[i_sample, :3] = np.matmul(R_i_g, np.matmul(R_i_g.T, acc_ori) + delta_acc[i_sample])
        return simulated_data


class CameraPlacementSimulator:
    def __init__(self, camera_param):
        self.intrinsic = np.array([[1500, 0., 540],
                                   [0., 1500, 945],
                                   [0., 0., 1.]])
        self.init_rot = camera_param['init_rot']
        self.init_tvec = camera_param['init_tvec']

    @staticmethod
    def _3d_marker_to_2d_pixel(data_3d, intrinsic, rot_mat, tvec):
        extrinsic_mat = np.concatenate([rot_mat, tvec.reshape([-1, 1])], axis=1)
        apply_fun = lambda data: np.matmul(intrinsic, np.matmul(extrinsic_mat, data))
        data_3d_homo = np.concatenate([data_3d, np.ones([data_3d.shape[0], 1])], axis=1)
        pixel_2d = np.apply_along_axis(apply_fun, 1, data_3d_homo)
        pixel_2d[:, 0] = pixel_2d[:, 0] / pixel_2d[:, 2]
        pixel_2d[:, 1] = pixel_2d[:, 1] / pixel_2d[:, 2]
        return pixel_2d[:, :2]

    def simulate_camera_orientation_and_position_delta_pixel(self, tvec_mm, rot_axis, rot_rad, marker_data):
        pixel_2d_ori = self._3d_marker_to_2d_pixel(marker_data, self.intrinsic, self.init_rot, self.init_tvec)

        # position
        new_tvec = self.init_tvec + tvec_mm
        pixel_2d_new = self._3d_marker_to_2d_pixel(marker_data, self.intrinsic, self.init_rot, new_tvec)
        delta_pixel_pos = pixel_2d_new - pixel_2d_ori

        # orientation
        rot_mat_additional = axangle2mat(rot_axis, rot_rad)
        new_rot = np.matmul(rot_mat_additional, self.init_rot)
        pixel_2d_new = self._3d_marker_to_2d_pixel(marker_data, self.intrinsic, new_rot, self.init_tvec)
        delta_pixel_ori = pixel_2d_new - pixel_2d_ori

        return delta_pixel_pos, delta_pixel_ori


def get_imu_loc_during_static(static_data, segment):
    center_of_markers_to_return = {
        'L_FOOT': ['LFM2'],
        'R_FOOT': ['RFM2'],
        'L_SHANK': ['LFME', 'LFAL'],
        'R_SHANK': ['RFME', 'RFAL'],
        'L_THIGH': ['LFME', 'LFT'],
        'R_THIGH': ['RFME', 'RFT'],
        'WAIST': ['LIAS', 'RIAS'],
        'CHEST': ['SXS', 'SJN']}
    marker_traj = [marker + axis for marker in center_of_markers_to_return[segment] for axis in ['_X', '_Y', '_Z']]
    if len(marker_traj) == 3:
        return static_data[marker_traj].values
    elif len(marker_traj) == 6:
        return (static_data[marker_traj[:3]].values + static_data[marker_traj[3:]].values) / 2


joint_2d_to_marker_3d = {
    "LShoulder": ['LAC', 'RAC'], "RShoulder": ['RAC'], "MidHip": ['LIPS', 'RIPS', 'LIAS', 'RIAS'],
    "RHip": ['RFT'], "LHip": ['LFT'], "RKnee": ['RFME', 'RFLE'], "LKnee": ['LFME', 'LFLE'],
    "RAnkle": ['RTAM', 'RFAL'], "LAnkle": ['LTAM', 'LFAL']}


def get_center_of_markers(data, markers):
    data_len = data.shape[0]
    average_data = np.zeros([data_len, 3])
    for marker in markers:
        marker_col = [marker + axis for axis in ['_X', '_Y', '_Z']]
        average_data += data[marker_col].values
    return average_data / len(markers)


def read_csv_as_float32(filename):
    df_test = pd.read_csv(filename, nrows=3)
    float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    df = pd.read_csv(filename, engine='c', dtype=float32_cols)
    return df


def generate_combined_data_placement_error():
    for subject in SUBJECTS[0:]:
        print('generate_combined_data_placement_error, {}'.format(subject))
        static_data = read_csv_as_float32(os.path.join(DATA_PATH, subject, "combined/static_back.csv"))
        static_data = pd.DataFrame(np.float32(np.mean(static_data.values, axis=0)).reshape([1, -1]), columns=static_data.columns)
        for trial in TRIALS:
            columns, data_with_error = [], []
            trial_data = read_csv_as_float32(os.path.join(DATA_PATH, subject, "combined", trial + ".csv"))

            for segment in SENSOR_LIST:
                imu_loc_during_static = get_imu_loc_during_static(static_data, segment)
                marker_cols = [marker + axis for marker in SEGMENT_DEFINITIONS[segment] for axis in ['_X', '_Y', '_Z']]
                static_marker_data = static_data[marker_cols].values
                trial_marker_data = trial_data[marker_cols].values
                imu_cols = [axis + '_' + segment for axis in IMU_FIELDS[:6]]
                imu_data = trial_data[imu_cols].values
                simulator = ImuPlacementSimulater(static_marker_data, trial_marker_data, imu_data)
                delta_acc_base, R_relative_to_static = simulator.get_delta_acc_and_R(imu_loc_during_static, y_mm=1)
                for magnitude_pos, magnitude_ori in zip(imu_error_magnitudes['_e_pos_y'], imu_error_magnitudes['_e_ori_z']):
                    for sign, sign_str in zip([1, -1], ['_+', '_-']):
                        imu_e_pos_y = simulator.add_delta_acc_to_data(delta_acc_base*sign*magnitude_pos, R_relative_to_static, R_i0_g=R_i0_g[segment])
                        imu_e_ori_z = simulator.simulate_orientation_error(z_rad=np.deg2rad(sign*magnitude_ori))
                        data_with_error.extend([imu_e_pos_y, imu_e_ori_z])
                        columns.extend([col + '_e_pos_y' + sign_str + str(magnitude_pos) for col in imu_cols])
                        columns.extend([col + '_e_ori_z' + sign_str + str(magnitude_ori) for col in imu_cols])

            for camera, camera_param in zip(['_90', '_180'], [params_90, params_180]):
                simulator = CameraPlacementSimulator(camera_param)
                for i_test in range(num_of_repeated_test):
                    rand_rot_axis = np.random.uniform(size=3)
                    rand_tvec = np.random.uniform(size=3)
                    rand_tvec = rand_tvec / np.linalg.norm(rand_tvec)
                    for joint in VIDEO_LIST[:-2]:
                        joint_column = [joint + axis + camera for axis in ['_x', '_y']]
                        joint_3d = get_center_of_markers(trial_data, joint_2d_to_marker_3d[joint])
                        for magnitude_pos, magnitude_ori in zip(camera_error_magnitudes['_e_pos'], camera_error_magnitudes['_e_ori']):
                            video_e_pos, video_e_ori = simulator.simulate_camera_orientation_and_position_delta_pixel(
                                magnitude_pos * rand_tvec, rand_rot_axis, np.deg2rad(magnitude_ori), joint_3d)
                            data_with_error.extend([video_e_pos, video_e_ori])
                            columns.extend([col + '_e_pos_' + str(magnitude_pos) + '_test' + str(i_test) for col in joint_column])
                            columns.extend([col + '_e_ori_' + str(magnitude_ori) + '_test' + str(i_test) for col in joint_column])

            error_df = pd.DataFrame(np.concatenate(data_with_error, axis=1), columns=columns)
            os.makedirs(os.path.join(DATA_PATH, subject, 'with_placement_error'), exist_ok=True)
            error_df.to_csv(os.path.join(DATA_PATH, subject, "with_placement_error", trial + ".csv"), index=False, float_format='%.5f')


def generate_step_data_placement_error():
    data_path = DATA_PATH + '/40samples+stance.h5'
    with h5py.File(data_path, 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items() if subject in SUBJECTS[0:]}
        data_fields = json.loads(hf.attrs['columns'])
    sample_index_col, trial_id_col = data_fields.index(SAMPLE_INDEX), data_fields.index(TRIAL_ID)
    with h5py.File(DATA_PATH + '/imu_with_placement_error.h5', 'w') as hf:
        for subject in SUBJECTS[0:]:
            print('generate_step_data_placement_error, ' + subject)
            sub_data = data_all_sub[subject]
            sub_placement_error_data_continuous = {i_trial: read_csv_as_float32(os.path.join(DATA_PATH, subject, "with_placement_error", trial + ".csv"))
                                                   for i_trial, trial in enumerate(TRIALS)}
            placement_error_cols = sub_placement_error_data_continuous[0].columns
            step_placement_error_data = np.zeros([sub_data.shape[0], sub_data.shape[1], len(placement_error_cols)], dtype='float32')
            for i_step in range(sub_data.shape[0]):
                step_index = sub_data[i_step, :, sample_index_col]
                trial_id = int(sub_data[i_step, 0, trial_id_col])
                start_index, end_index = int(step_index[0]), int(max(step_index) + 1)
                sub_placement_error_data_continuous_trial = sub_placement_error_data_continuous[trial_id]
                step_placement_error_data[i_step, :end_index-start_index, :] = sub_placement_error_data_continuous_trial.iloc[start_index:end_index]

            hf.create_dataset(subject, data=step_placement_error_data, dtype='float32')
        hf.attrs['columns'] = json.dumps(list(placement_error_cols))

    # placement_error_data_all_sub = {}
    #     placement_error_data_all_sub[subject] = step_placement_error_data
    #
    # with h5py.File(DATA_PATH + '/imu_with_placement_error.h5', 'w') as hf:
    #     for subject, sub_data in placement_error_data_all_sub.items():
    #         hf.create_dataset(subject, data=sub_data, dtype='float32')
    #     hf.attrs['columns'] = json.dumps(list(placement_error_cols))


def replace_data_or_add_delta_and_pred(is_replace, data_fields, loc_error_free, loc_with_error,
                                       subject_data, subject_data_error, model, model_inputs):
    vid_fields, i_high_level_start = VID_ALL, 0
    subject_data = np.array(subject_data, copy=True)
    if is_replace:
        subject_data[:, :, loc_error_free] = subject_data_error[:, :, loc_with_error]
    else:
        subject_data[:, :, loc_error_free] += subject_data_error[:, :, loc_with_error]
    antro_data = subject_data[:, :, [data_fields.index(field) for field in STATIC_DATA]]
    high_level_data = subject_data[:, :, [data_fields.index(field) for field in HIGH_LEVEL_FEATURE]]
    high_level_data = normalize_array_separately(high_level_data, model.scalars['high_level'], 'transform')
    model_inputs['others'] = torch.from_numpy(np.concatenate([antro_data, high_level_data[:, :, i_high_level_start:]], axis=2)).float().cuda()

    subject_data = make_vid_relative_to_midhip(subject_data, data_fields)
    subject_data = normalize_vid_by_size_of_subject_in_static_trial(subject_data, data_fields)

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
    trials.append(mean_series.append(pd.DataFrame(trials)[[x for x in trials[0].keys() if x not in mean_series.keys()]].mean()))
    return trials


def replace_data_and_test():
    all_trials = []
    for subject in SUBJECTS[0:]:
        with h5py.File(DATA_PATH + '/40samples+stance.h5', 'r') as hf:
            data_all_sub = {subject: subject_data[:] for subject_load, subject_data in hf.items() if subject_load == subject}
            data_fields = json.loads(hf.attrs['columns'])
        with h5py.File(DATA_PATH + '/imu_with_placement_error.h5', 'r') as hf:
            data_all_sub_error = {subject: subject_data[:] for subject_load, subject_data in hf.items() if subject_load == subject}
            data_fields_error = json.loads(hf.attrs['columns'])

        print('replace_data_and_test, ' + subject)
        model_path = os.path.join('..', 'figures', 'results', '1115', 'TfnNet', 'sub_models', subject, 'model.pth')
        model = torch.load(model_path).cuda()

        subject_data, subject_data_error = data_all_sub[subject], data_all_sub_error[subject]
        ground_truth_moment = subject_data[:, :, [data_fields.index(field) for field in ['EXT_KM_X', 'EXT_KM_Y']]]
        weights = subject_data[:, :, [data_fields.index(FORCE_PHASE), data_fields.index(FORCE_PHASE)]]
        model_inputs = {'step_length': torch.from_numpy(np.sum(~(subject_data[:, :, 0] == 0.), axis=1))}

        """ IMU placements """
        for error_name in imu_error_names:
            for magnitude in imu_error_magnitudes[error_name]:
                for i_test in range(num_of_repeated_test):
                    rand_sign_list = [random.choice(['+', '-']) for i in range(len(SENSOR_LIST))]
                    imu_loc_error_free = [data_fields.index(axis + '_' + sensor) for sensor in SENSOR_LIST for axis in IMU_FIELDS[:6]]
                    imu_col_with_error = [axis + '_' + sensor + error_name + '_' + rand_sign_list[i_sensor] + str(magnitude)
                                          for i_sensor, sensor in enumerate(SENSOR_LIST) for axis in IMU_FIELDS[:6]]
                    imu_loc_with_error = [data_fields_error.index(col) for col in imu_col_with_error]
                    predicted = replace_data_or_add_delta_and_pred(True, data_fields, imu_loc_error_free, imu_loc_with_error,
                                                                   subject_data, subject_data_error, model, model_inputs)
                    param_to_log = {'subject': subject, 'model_name': 'TfnNet', 'error_sensor': 'IMU',
                                    'error_name': error_name, 'error_magnitude': magnitude, 'test': i_test}
                    all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted,
                                                      weights, param_to_log))

        """ Camera placements """
        for error_name in camera_error_names:
            camera_col_error_free = [joint + axis + camera for camera in ['_90', '_180'] for joint in VIDEO_LIST[:-2] for axis in ['_x', '_y']]
            camera_loc_error_free = [data_fields.index(col) for col in camera_col_error_free]
            for magnitude in camera_error_magnitudes[error_name]:
                for i_test in range(num_of_repeated_test):
                    camera_col_with_error = [joint + axis + camera + error_name + '_' + str(magnitude) + '_test' + str(i_test)
                                             for camera in ['_90', '_180'] for joint in VIDEO_LIST[:-2] for axis in ['_x', '_y']]
                    camera_loc_with_error = [data_fields_error.index(col) for col in camera_col_with_error]
                    predicted = replace_data_or_add_delta_and_pred(False, data_fields, camera_loc_error_free, camera_loc_with_error,
                                                                   subject_data, subject_data_error, model, model_inputs)
                    param_to_log = {'subject': subject, 'model_name': 'TfnNet', 'error_sensor': 'camera',
                                    'error_name': error_name, 'error_magnitude': magnitude, 'test': i_test}
                    all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted,
                                                      weights, param_to_log))

    results_df = pd.DataFrame(all_trials)
    results_df.to_csv('error_margin_results.csv', index=False)


def plot_result_lines():
    def save_fig(name, dpi=300):
        plt.savefig('../figures/exports/' + name + '.png', dpi=dpi)

    def get_results_of_one_condition(error_sensor, error_name, moment_name):
        condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_sensor'] == error_sensor)
                                  & (results_df['error_name'] == error_name)]
        results = []
        if error_sensor == 'IMU':
            error_magnitudes = imu_error_magnitudes
        else:
            error_magnitudes = camera_error_magnitudes
        for magnitude in error_magnitudes[error_name]:
            test_results = []
            for i_test in range(num_of_repeated_test):
                test_df = condition_df[(condition_df['error_magnitude'] == magnitude) & (condition_df['test'] == i_test)]
                test_results.append(np.mean(test_df['RMSE_' + moment_name]))
            results.append([np.mean(test_results), np.min(test_results), np.max(test_results)])
        results.insert(0, [no_err_accuracy[moment_name] for i in range(3)])
        return np.array(results)

    def draw_subplot(result_kam, result_kfm, ax, magnitudes_plot, x_label):
        def curve_of_one_moment(result, color):
            lower_diff, upper_diff = result[:, 0] - result[:, 1], result[:, 2] - result[:, 0]
            plt.errorbar(axis_x, result[:, 0], yerr=[lower_diff, upper_diff], linewidth=LINE_WIDTH, elinewidth=LINE_WIDTH, label='KAM', color=color)
            ax.fill_between(axis_x, result[:, 1], result[:, 2], facecolor=color, alpha=0.4)

        axis_x = [0] + magnitudes_plot
        color_0, color_1 = np.array([90, 140, 20]) / 255, np.array([0, 103, 137]) / 255
        curve_of_one_moment(result_kam, color_0)
        curve_of_one_moment(result_kfm, color_1)
        ax.tick_params(labelsize=FONT_DICT_SMALL['fontsize'])
        ax.set_xlim(0, max(axis_x) + 0.2)
        ax.set_xticks(axis_x)
        ax.set_xticklabels(axis_x, fontdict=FONT_DICT_SMALL)
        ax.set_xlabel(x_label, fontdict=FONT_DICT_SMALL)
        max_y = max(np.max(result_kam[:, 2]), np.max(result_kfm[:, 2]))
        y_lim = np.ceil(max_y * 10) / 10
        ax.set_ylim(0, y_lim)
        y_tick_locs = np.arange(0, y_lim+0.2, 0.4)
        ax.set_yticks(y_tick_locs)
        ax.set_yticklabels([round(x, 1) for x in y_tick_locs], fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('RMSE', fontdict=FONT_DICT_SMALL)
        format_axis()

    results_df = read_csv_as_float32('error_margin_results.csv')
    rc('font', family='Arial')
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 6, 6])

    imu_e_pos_kam, imu_e_pos_kfm = get_results_of_one_condition('IMU', '_e_pos_y', 'KAM'), get_results_of_one_condition('IMU', '_e_pos_y', 'KFM')
    draw_subplot(imu_e_pos_kam, imu_e_pos_kfm, fig.add_subplot(gs[1, 0]), imu_error_magnitudes['_e_pos_y'], 'IMU Position Change (mm)')

    imu_e_ori_kam, imu_e_ori_kfm = get_results_of_one_condition('IMU', '_e_ori_z', 'KAM'), get_results_of_one_condition('IMU', '_e_ori_z', 'KFM')
    draw_subplot(imu_e_ori_kam, imu_e_ori_kfm, fig.add_subplot(gs[1, 1]), imu_error_magnitudes['_e_ori_z'], 'IMU Orientation Change (deg)')

    camera_e_pos_kam, camera_e_pos_kfm = get_results_of_one_condition('camera', '_e_pos', 'KAM'), get_results_of_one_condition('camera', '_e_pos', 'KFM')
    draw_subplot(camera_e_pos_kam, camera_e_pos_kfm, fig.add_subplot(gs[2, 0]), camera_error_magnitudes['_e_pos'], 'Camera Position Change (mm)')

    camera_e_ori_kam, camera_e_ori_kfm = get_results_of_one_condition('camera', '_e_ori', 'KAM'), get_results_of_one_condition('camera', '_e_ori', 'KFM')
    draw_subplot(camera_e_ori_kam, camera_e_ori_kfm, fig.add_subplot(gs[2, 1]), camera_error_magnitudes['_e_ori'], 'Camera Orientation Change (deg)')

    plt.tight_layout()
    save_fig('error_margin')
    plt.show()


def plot_result_boxes():
    ylims_all = {'KAM': {'IMU': {'_e_pos_y': [0.48, 0.6, 0.03], '_e_ori_z': [0.45, 0.7, 0.05]},
                         'camera': {'_e_pos': [0.49, 0.55, 0.02], '_e_ori': [0.47, 0.715, 0.06]}},
                 'KFM': {'IMU': {'_e_pos_y': [0.63, 0.87, 0.06], '_e_ori_z': [0.63, 0.91, 0.07]},
                         'camera': {'_e_pos': [0.65, 0.7, 0.01], '_e_ori': [0.6, 1.42, 0.2]}}}
    def save_fig(name, dpi=300):
        plt.savefig('../figures/exports/' + name + '.png', dpi=dpi)

    def get_results_of_one_condition(error_sensor, error_name, moment_name):
        condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_sensor'] == error_sensor)
                                  & (results_df['error_name'] == error_name)]
        base_diff = {'KAM': 0.000869481595430821, 'KFM': 0.014753654202022015}
        results = []
        if error_sensor == 'IMU':
            error_magnitudes = imu_error_magnitudes
        else:
            error_magnitudes = camera_error_magnitudes
        for magnitude in error_magnitudes[error_name]:
            test_results = []
            for i_test in range(num_of_repeated_test):
                test_df = condition_df[(condition_df['error_magnitude'] == magnitude) & (condition_df['test'] == i_test)]
                test_results.append(np.mean(test_df['RMSE_' + moment_name]))
            results.append(test_results)
        return np.array(results) + base_diff[moment_name]

    def draw_subplot(result, ax, magnitudes_plot, x_label, ylims):
        plt.plot([-0.5, result.shape[0]-0.5], [no_err_accuracy[moment], no_err_accuracy[moment]], '--', color='gray', linewidth=LINE_WIDTH)
        for i_err in range(result.shape[0]):
            box_ = plt.boxplot(result[i_err], positions=[i_err], widths=[0.5], patch_artist=True)
            for field in ['medians', 'whiskers', 'caps', 'boxes']:
                [box_[field][i].set(linewidth=LINE_WIDTH) for i in range(len(box_[field]))]
            [box_['fliers'][i].set(marker='o', markeredgecolor='black', markersize=7) for i in range(len(box_['fliers']))]
            box_['medians'][0].set(linewidth=LINE_WIDTH, color=[1, 1, 1])
        ax.tick_params(labelsize=FONT_DICT_SMALL['fontsize'])
        ax.set_xlim(-0.5, result.shape[0]-0.5)
        ax.set_xticks(range(result.shape[0]))
        ax.set_xticklabels(magnitudes_plot, fontdict=FONT_DICT_SMALL)
        ax.set_xlabel(x_label, fontdict=FONT_DICT_SMALL)
        # max_y = max(np.max(result_kam[:, 2]), np.max(result_kfm[:, 2]))
        ax.set_ylim(ylims[0], ylims[1])
        y_tick_locs = np.arange(ylims[0], ylims[1]+0.001, ylims[2])
        ax.set_yticks(y_tick_locs)
        ax.set_yticklabels([round(x, 2) for x in y_tick_locs], fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('RMSE of {} (%BW$\cdot$BH)'.format(moment), fontdict=FONT_DICT_SMALL)
        format_axis()

    results_df = read_csv_as_float32('error_margin_results.csv')
    rc('font', family='Arial')
    for moment in ['KAM', 'KFM']:
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1])

        imu_e_pos = get_results_of_one_condition('IMU', '_e_pos_y', moment)
        draw_subplot(imu_e_pos, fig.add_subplot(gs[0, 0]), imu_error_magnitudes['_e_pos_y'],
                     'IMU Position Variation (mm)', ylims_all[moment]['IMU']['_e_pos_y'])
        imu_e_ori = get_results_of_one_condition('IMU', '_e_ori_z', moment)
        draw_subplot(imu_e_ori, fig.add_subplot(gs[0, 1]), imu_error_magnitudes['_e_ori_z'],
                     'IMU Orientation Variation (deg)', ylims_all[moment]['IMU']['_e_ori_z'])
        camera_e_pos = get_results_of_one_condition('camera', '_e_pos', moment)
        draw_subplot(camera_e_pos, fig.add_subplot(gs[1, 0]), camera_error_magnitudes['_e_pos'],
                     'Camera Position Variation (mm)', ylims_all[moment]['camera']['_e_pos'])
        camera_e_ori = get_results_of_one_condition('camera', '_e_ori', moment)
        draw_subplot(camera_e_ori, fig.add_subplot(gs[1, 1]), camera_error_magnitudes['_e_ori'],
                     'Camera Orientation Variation (deg)', ylims_all[moment]['camera']['_e_ori'])

        plt.tight_layout(w_pad=4, h_pad=4)
        save_fig('error_margin_' + moment)
    plt.show()


num_of_repeated_test = 10
no_err_accuracy = {'KAM': 0.4935584345953873, 'KFM': 0.6591598856509528}
imu_error_magnitudes = {'_e_pos_y': [50, 100, 150, 200], '_e_ori_z': [5, 10, 15, 20]}
imu_error_names = list(imu_error_magnitudes.keys())
camera_error_magnitudes = {'_e_pos': [100, 200, 300, 400], '_e_ori': [5, 10, 15, 20]}
camera_error_names = list(camera_error_magnitudes.keys())

if __name__ == "__main__":
    # generate_combined_data_placement_error()
    # generate_step_data_placement_error()
    # replace_data_and_test()
    plot_result_boxes()

