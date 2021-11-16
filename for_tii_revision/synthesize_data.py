from const import DATA_PATH, SUBJECTS, TRIALS, VICON_SAMPLE_RATE, SEGMENT_DEFINITIONS, SENSOR_LIST, IMU_FIELDS, \
    SAMPLE_INDEX, TRIAL_ID
import os
import pandas as pd
import numpy as np
import torch
import scipy.interpolate as interpo
from scipy.stats import sem
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
from wearable_toolkit import data_filter
import h5py
import json
from alan_framework import TfnNet, InertialNet, OutNet, VideoNet, ACC_ALL, GYR_ALL, VID_ALL, BaseFramework
from const import STATIC_DATA, HIGH_LEVEL_FEATURE, FORCE_PHASE, USED_KEYPOINTS, SUBJECT_HEIGHT, VIDEO_LIST
from figures.table_metrics import get_score, append_mean_results_in_the_end, get_metric_mean_std_result
from a_load_model_and_predict import normalize_array_separately


R_i0_g = {
    'CHEST': np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    'WAIST': np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    'L_THIGH': np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    'R_THIGH': np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    'L_SHANK': np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    'R_SHANK': np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    'L_FOOT': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    'R_FOOT': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
}


def make_vid_relative_to_midhip(sub_data, data_fields):
    midhip_col_loc = [data_fields.index('MidHip' + axis + angle) for axis in ['_x', '_y'] for angle in
                      ['_90', '_180']]
    midhip_90_and_180_data = sub_data[:, :, midhip_col_loc]
    for key_point in USED_KEYPOINTS:
        key_point_col_loc = [data_fields.index(key_point + axis + angle) for axis in ['_x', '_y'] for angle in
                             ['_90', '_180']]
        sub_data[:, :, key_point_col_loc] = sub_data[:, :, key_point_col_loc] - midhip_90_and_180_data
    return sub_data


def normalize_vid_by_size_of_subject_in_static_trial(sub_data, data_fields):
    height_col_loc = data_fields.index(SUBJECT_HEIGHT)
    sub_height = sub_data[0, 0, height_col_loc]
    for camera in ['90', '180']:
        vid_col_loc = [data_fields.index(keypoint + axis + camera) for keypoint in USED_KEYPOINTS for axis in
                       ['_x_', '_y_']]
        sub_data[:, :, vid_col_loc] = sub_data[:, :, vid_col_loc] / sub_height
    return sub_data


# def get_virtual_marker_trajectory(virtual_marker_center_at_static_pose, segment_marker_during_walking,
#                                   segment_marker_at_static_pose, R_static_pose_to_ground):
#     segment_marker_num = segment_marker_at_static_pose.shape[0]
#     segment_marker_during_walking = segment_marker_during_walking.as_matrix()
#     data_len = segment_marker_during_walking.shape[0]
#     virtual_marker = np.zeros([data_len, 3])
#     R_IMU_transform = np.zeros([3, 3, data_len])
#     for i_frame in range(data_len):
#         current_marker_matrix = segment_marker_during_walking[i_frame, :].reshape([segment_marker_num, 3])
#         [R_between_frames, t] = _rigid_transform_3D(segment_marker_at_static_pose, current_marker_matrix)
#         virtual_marker[i_frame, :] = (np.dot(R_between_frames, virtual_marker_center_at_static_pose) + t)
#         R_IMU_transform[:, :, i_frame] = np.matmul(R_static_pose_to_ground, R_between_frames.T)
#     return virtual_marker, R_IMU_transform



class Simulator:
    def __init__(self, marker_at_static_pose, marker_during_walking, imu_data):
        self.marker_at_static_pose = marker_at_static_pose
        self.marker_during_walking = marker_during_walking
        self.imu_data = imu_data
        # self.imu_loc_at_static_pose = imu_loc_during_static

    @staticmethod
    def _rigid_transform_3D(A, B):
        assert len(A) == len(B)
        N = A.shape[0]  # total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        # dot is matrix multiplication for array
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = - np.dot(R, centroid_A.T) + centroid_B.T
        return R, t

    def __simulate_virtual_marker_trajectory(self, imu_loc_during_static):
        def apply_fun(current_marker):
            current_marker_matrix = current_marker.reshape([segment_marker_num, 3])
            [R_between_frames, t] = Simulator._rigid_transform_3D(
                segment_marker_at_static_pose, current_marker_matrix)
            virtual_marker = (np.dot(R_between_frames, imu_loc_during_static.T).T + t)
            return np.concatenate([R_between_frames, virtual_marker])
        segment_marker_at_static_pose = self.marker_at_static_pose.reshape([-1, 3])
        segment_marker_num = segment_marker_at_static_pose.shape[0]
        R_and_marker = np.apply_along_axis(apply_fun, 1, self.marker_during_walking)
        R_relative_to_static, virtual_marker = R_and_marker[:, :3], R_and_marker[:, 3]
        return R_relative_to_static, virtual_marker

    def simulate_orientation_error(self, x_rad=0., y_rad=0., z_rad=0.):
        """
        :param x_rad: float, orientation error over x axis in radius
        :param y_rad: float, orientation error over y axis in radius
        :param z_rad: float, orientation error over z axis in radius
        :return: imu_data_with_orientation_error
        """
        def apply_fun(data):
            data = np.matmul(rot_mat, data)
            return data
        rot_mat = euler2mat(x_rad, y_rad, z_rad)
        simulated_data = np.array(self.imu_data, copy=True)
        simulated_data[:, :3] = np.apply_along_axis(apply_fun, 1, simulated_data[:, :3])
        simulated_data[:, 3:] = np.apply_along_axis(apply_fun, 1, simulated_data[:, 3:])
        # for i in range(6):
        #     plt.figure()
        #     plt.plot(self.imu_data[:, i])
        #     plt.plot(simulated_data[:, i])
        # plt.show()
        return simulated_data

    def simulate_position_error(self, imu_loc_during_static, R_i0_g=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), x_mm=0., y_mm=0., z_mm=0.):
        """
        :param r_imu_earth:
        :param x_mm: float, position error over x axis in mm at sample 0
        :param y_mm: float, position error over y axis in mm at sample 0
        :param z_mm: float, position error over z axis in mm at sample 0
        :return: imu_data_with_position_error
        """
        R_relative_to_static, imu_trajectory = self.__simulate_virtual_marker_trajectory(imu_loc_during_static)
        _, imu_trajectory_with_position_error = self.__simulate_virtual_marker_trajectory(imu_loc_during_static + np.array([x_mm, y_mm, z_mm]))
        delta_acc = self.__get_delta_acc_in_earth_frame(imu_trajectory, imu_trajectory_with_position_error)
        simulated_data = np.array(self.imu_data, copy=True)
        data_len = R_relative_to_static.shape[0]
        for i_sample in range(data_len):
            R_i_g = np.matmul(R_relative_to_static[i_sample], R_i0_g)
            acc_ori = simulated_data[i_sample, :3]
            simulated_data[i_sample, :3] = np.matmul(R_i_g, np.matmul(R_i_g.T, acc_ori) + delta_acc[i_sample])
        return simulated_data

    @staticmethod
    def __get_delta_acc_in_earth_frame(imu_trajectory, imu_trajectory_with_position_error):
        imu_trajectory, imu_trajectory_with_position_error = imu_trajectory * 1e-3, imu_trajectory_with_position_error * 1e-3
        data_len = imu_trajectory.shape[0]
        step_marker = np.arange(0, data_len / VICON_SAMPLE_RATE - 1e-12, 1 / VICON_SAMPLE_RATE)
        tck, step_marker = interpo.splprep(imu_trajectory.T, u=step_marker, s=0)
        acc_center = np.column_stack(interpo.splev(step_marker, tck, der=2))  # der=2 means take the second derivation
        tck, step_marker = interpo.splprep(imu_trajectory_with_position_error.T, u=step_marker, s=0)
        acc_with_placement_error = np.column_stack(interpo.splev(step_marker, tck, der=2))
        delta_acc = acc_center - acc_with_placement_error
        delta_acc_filtered = data_filter(delta_acc, 15, VICON_SAMPLE_RATE)
        return delta_acc_filtered


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


def print_t_IV_report_absolute_RMSE():
    results_df = pd.read_csv('placement_error_results.csv')
    metric_name = 'RMSE_'
    """ No error """
    condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_type'] == 'no')]
    for target in ['KAM', 'KFM']:
        print('&{:6.2f} ({:3.2f})'.format(condition_df[metric_name+target].mean(), condition_df[metric_name+target].sem()), end='\t')
    print()

    """ Single IMU position """
    for target in ['KAM', 'KFM']:
        mean_, sem_ = 0, 0
        for error_name in error_names[:2]:
            for sensor in SENSOR_LIST:
                condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == sensor) &
                                          (results_df['error_name'] == error_name)]
                if np.mean(condition_df[metric_name + target]) > mean_:
                    mean_ = condition_df[metric_name + target].mean()
                    sem_ = condition_df[metric_name + target].sem()
        print('&{:6.2f} ({:3.2f})'.format(mean_, sem_), end='\t')
    print()

    """ Single orientation """
    for target in ['KAM', 'KFM']:
        mean_, sem_ = 0, 0
        for sensor in SENSOR_LIST:
            condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == sensor) &
                                      (results_df['error_name'] == error_names[2])]
            if np.mean(condition_df[metric_name + target]) > mean_:
                mean_ = condition_df[metric_name + target].mean()
                sem_ = condition_df[metric_name + target].sem()
        print('&{:6.2f} ({:3.2f})'.format(mean_, sem_), end='\t')
    print()

    """ Multiple simultaneous position """
    for target in ['KAM', 'KFM']:
        mean_, sem_ = 0, 0
        for error_name in error_names[:2]:
            condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == 'all') &
                                      (results_df['error_name'] == error_name)]
            if np.mean(condition_df[metric_name + target]) > mean_:
                mean_ = condition_df[metric_name + target].mean()
                sem_ = condition_df[metric_name + target].sem()
        print('&{:6.2f} ({:3.2f})'.format(mean_, sem_), end='\t')
    print()

    """ Multiple simultaneous orientation """
    for target in ['KAM', 'KFM']:
        condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == 'all') &
                                  (results_df['error_name'] == error_names[2])]
        print('&{:6.2f} ({:3.2f})'.format(condition_df[metric_name+target].mean(), condition_df[metric_name+target].sem()), end='\t')
    print()


def generate_combined_data_placement_error():
    for subject in SUBJECTS:
        print('generate_combined_data_placement_error, {}'.format(subject))
        static_data = pd.read_csv(os.path.join(DATA_PATH, subject, "combined/static_back.csv"))
        static_data = pd.DataFrame(np.mean(static_data.values, axis=0).reshape([1, -1]), columns=static_data.columns)
        for trial in TRIALS:
            columns, data_with_error = [], []
            trial_data = pd.read_csv(os.path.join(DATA_PATH, subject, "combined", trial + ".csv"))
            for segment in SENSOR_LIST:
                imu_loc_during_static = get_imu_loc_during_static(static_data, segment)
                marker_cols = [marker + axis for marker in SEGMENT_DEFINITIONS[segment] for axis in ['_X', '_Y', '_Z']]
                static_marker_data = static_data[marker_cols].values
                trial_marker_data = trial_data[marker_cols].values
                imu_cols = [axis + '_' + segment for axis in IMU_FIELDS[:6]]
                imu_data = trial_data[imu_cols].values
                simulator = Simulator(static_marker_data, trial_marker_data, imu_data)
                imu_e_ori_z = simulator.simulate_orientation_error(z_rad=np.deg2rad(-10))
                imu_e_pos_x = simulator.simulate_position_error(imu_loc_during_static, R_i0_g=R_i0_g[segment], x_mm=100)
                imu_e_pos_y = simulator.simulate_position_error(imu_loc_during_static, R_i0_g=R_i0_g[segment], y_mm=100)
                data_with_error.extend([imu_e_pos_x, imu_e_pos_y, imu_e_ori_z])
                columns.extend([col + error_name for error_name in error_names for col in imu_cols])
            error_df = pd.DataFrame(np.concatenate(data_with_error, axis=1), columns=columns)
            os.makedirs(os.path.join(DATA_PATH, subject, 'with_placement_error'), exist_ok=True)
            error_df.to_csv(os.path.join(DATA_PATH, subject, "with_placement_error", trial + ".csv"), index=False)


def generate_step_data_placement_error():
    print(generate_step_data_placement_error)
    data_path = DATA_PATH + '/40samples+stance.h5'
    with h5py.File(data_path, 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])
    sample_index_col, trial_id_col = data_fields.index(SAMPLE_INDEX), data_fields.index(TRIAL_ID)
    placement_error_data_all_sub = {}
    for subject in SUBJECTS:
        sub_data = data_all_sub[subject]
        sub_placement_error_data_continuous = {i_trial: pd.read_csv(os.path.join(DATA_PATH, subject, "with_placement_error", trial + ".csv"))
                                               for i_trial, trial in enumerate(TRIALS)}
        placement_error_cols = sub_placement_error_data_continuous[0].columns
        step_placement_error_data = np.zeros([sub_data.shape[0], sub_data.shape[1], len(placement_error_cols)])
        for i_step in range(sub_data.shape[0]):
            step_index = sub_data[i_step, :, sample_index_col]
            trial_id = int(sub_data[i_step, 0, trial_id_col])
            start_index, end_index = int(step_index[0]), int(max(step_index) + 1)
            sub_placement_error_data_continuous_trial = sub_placement_error_data_continuous[trial_id]
            step_placement_error_data[i_step, :end_index-start_index, :] = sub_placement_error_data_continuous_trial.iloc[start_index:end_index]
            x=1
        placement_error_data_all_sub[subject] = step_placement_error_data

    with h5py.File(DATA_PATH + '/imu_with_placement_error.h5', 'w') as hf:
        for subject, sub_data in placement_error_data_all_sub.items():
            hf.create_dataset(subject, data=sub_data, dtype='float32')
        hf.attrs['columns'] = json.dumps(list(placement_error_cols))


def replace_data_and_pred(data_fields, imu_loc_error_free, imu_loc_with_error,
                          subject_data, subject_data_error, model, model_inputs):
    subject_data = np.array(subject_data, copy=True)
    subject_data[:, :, imu_loc_error_free] = subject_data_error[:, :, imu_loc_with_error]
    antro_data = subject_data[:, :, [data_fields.index(field) for field in STATIC_DATA]]
    high_level_data = subject_data[:, :, [data_fields.index(field) for field in HIGH_LEVEL_FEATURE]]
    high_level_data = normalize_array_separately(high_level_data, model.scalars['high_level'], 'transform')
    model_inputs['others'] = torch.from_numpy(np.concatenate([antro_data, high_level_data], axis=2)).float().cuda()

    for name, fields in zip(['input_acc', 'input_gyr', 'input_vid'], [ACC_ALL, GYR_ALL, VID_ALL]):
        data = subject_data[:, :, [data_fields.index(field) for field in fields]]
        data = normalize_array_separately(data, model.scalars[name], 'transform')
        model_inputs[name] = torch.from_numpy(data).float().cuda()
    # model_inputs['input_acc'] = torch.from_numpy(
    #     subject_data[:, :, [data_fields.index(field) for field in ACC_ALL]]).float().cuda()
    # model_inputs['input_gyr'] = torch.from_numpy(
    #     subject_data[:, :, [data_fields.index(field) for field in GYR_ALL]]).float().cuda()
    # model_inputs['input_vid'] = torch.from_numpy(
    #     subject_data[:, :, [data_fields.index(field) for field in VID_ALL]]).float().cuda()
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
    print(replace_data_and_test)
    with h5py.File(DATA_PATH + '/40samples+stance.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])
    with h5py.File(DATA_PATH + '/imu_with_placement_error.h5', 'r') as hf:
        data_all_sub_error = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields_error = json.loads(hf.attrs['columns'])

    all_trials = []
    for subject in SUBJECTS:
        model_path = os.path.join('..', 'figures', 'results', '1115', 'TfnNet', 'sub_models', subject, 'model.pth')
        model = torch.load(model_path).cuda()

        subject_data, subject_data_error = data_all_sub[subject], data_all_sub_error[subject]
        ground_truth_moment = subject_data[:, :, [data_fields.index(field) for field in ['EXT_KM_X', 'EXT_KM_Y']]]
        # evaluate_fields = {'main_output': ['EXT_KM_X', 'EXT_KM_Y']}
        weights = subject_data[:, :, [data_fields.index(FORCE_PHASE), data_fields.index(FORCE_PHASE)]]
        model_inputs = {'step_length': torch.from_numpy(np.sum(~(subject_data[:, :, 0] == 0.), axis=1))}
        subject_data = make_vid_relative_to_midhip(subject_data, data_fields)
        subject_data = normalize_vid_by_size_of_subject_in_static_trial(subject_data, data_fields)

        """ No error """
        predicted = replace_data_and_pred(data_fields, [], [], subject_data, subject_data_error, model, model_inputs)
        param_to_log = {'subject': subject, 'error_type': 'no', 'error_segment': 'na', 'error_name': 'na'}
        all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted, weights, param_to_log))

        """ Single IMU has error """
        for error_name in error_names:
            for sensor in SENSOR_LIST:
                imu_loc_error_free = [data_fields.index(axis + '_' + sensor) for axis in IMU_FIELDS[:6]]
                imu_loc_with_error = [data_fields_error.index(axis + '_' + sensor + error_name) for axis in IMU_FIELDS[:6]]
                predicted = replace_data_and_pred(data_fields, imu_loc_error_free, imu_loc_with_error,
                                                  subject_data, subject_data_error, model, model_inputs)
                param_to_log = {'subject': subject, 'error_type': 'single', 'error_segment': sensor, 'error_name': error_name}
                all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted,
                                                  weights, param_to_log))

        """ Multiple simultaneous IMUs have error """
        for error_name in error_names:
            imu_loc_error_free = [data_fields.index(axis + '_' + sensor) for sensor in SENSOR_LIST for axis in IMU_FIELDS[:6]]
            imu_loc_with_error = [data_fields_error.index(axis + '_' + sensor + error_name) for sensor in SENSOR_LIST for axis in IMU_FIELDS[:6]]
            predicted = replace_data_and_pred(data_fields, imu_loc_error_free, imu_loc_with_error,
                                              subject_data, subject_data_error, model, model_inputs)
            param_to_log = {'subject': subject, 'error_type': 'multiple', 'error_segment': 'all', 'error_name': error_name}
            all_trials.extend(get_all_results(subject_data, data_fields, ground_truth_moment, predicted,
                                              weights, param_to_log))

    results_df = pd.DataFrame(all_trials)
    results_df.to_csv('placement_error_results.csv', index=False)


def print_t_IV_report_increase_of_RMSE():
    results_df = pd.read_csv('placement_error_results.csv')
    metric_name = 'rRMSE_'
    """ No error """
    no_error_df = results_df[(results_df['trial'] == 'all') & (results_df['error_type'] == 'no')]

    """ Single IMU position """
    print('\multirow{2}{*}{Single IMU}\t& Position Errors', end='\t')
    for target in ['KAM', 'KFM']:
        mean_, sem_ = 0, 0
        for error_name in error_names[:2]:
            for sensor in SENSOR_LIST:
                condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == sensor) &
                                          (results_df['error_name'] == error_name)]
                increases_ = condition_df[metric_name + target].values - no_error_df[metric_name + target].values
                if np.mean(increases_) > mean_:
                    mean_, sem_ = np.mean(increases_), sem(increases_)
        print('&{:6.2f} ({:3.2f})'.format(mean_, sem_), end='\t')
    print('\\\\')

    """ Single orientation """
    print('& Orientation Errors ', end='\t')
    for target in ['KAM', 'KFM']:
        mean_, sem_ = 0, 0
        for sensor in SENSOR_LIST:
            condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == sensor) &
                                      (results_df['error_name'] == error_names[2])]
            increases_ = condition_df[metric_name + target].values - no_error_df[metric_name + target].values
            if np.mean(increases_) > mean_:
                mean_, sem_ = np.mean(increases_), sem(increases_)
        print('&{:6.2f} ({:3.2f})'.format(mean_, sem_), end='\t')
    print('\\\\')
    print('\cmidrule{1-4}')

    """ Multiple simultaneous position """
    print('\multirow{2}{*}{All IMUs}\t& Position Errors', end='\t')
    for target in ['KAM', 'KFM']:
        mean_, sem_ = 0, 0
        for error_name in error_names[:2]:
            condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == 'all') &
                                      (results_df['error_name'] == error_name)]
            increases_ = condition_df[metric_name + target].values - no_error_df[metric_name + target].values
            if np.mean(increases_) > mean_:
                mean_, sem_ = np.mean(increases_), sem(increases_)
        print('&{:6.2f} ({:3.2f})'.format(mean_, sem_), end='\t')
    print('\\\\')

    """ Multiple simultaneous orientation """
    print('& Orientation Errors ', end='\t')
    for target in ['KAM', 'KFM']:
        condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == 'all') &
                                  (results_df['error_name'] == error_names[2])]
        increases_ = condition_df[metric_name + target].values - no_error_df[metric_name + target].values
        print('&{:6.2f} ({:3.2f})'.format(np.mean(increases_), sem(increases_)), end='\t')
    print('\\\\')


error_names = ['_e_pos_x', '_e_pos_y', '_e_ori_z']

if __name__ == "__main__":
    # generate_combined_data_placement_error()
    # generate_step_data_placement_error()
    # replace_data_and_test()
    # print_t_IV_report_absolute_RMSE()
    print_t_IV_report_increase_of_RMSE()


