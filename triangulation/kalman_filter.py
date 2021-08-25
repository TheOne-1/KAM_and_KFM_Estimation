import numpy as np
from const import SUBJECTS, GRAVITY, TRIALS, STATIC_TRIALS
from triangulation.triangulation_toolkit import init_kalman_param_static, q_to_knee_angle, \
    compare_axes_results, plot_q_for_debug, figure_for_FE_AA_angles, print_h_mat, compare_three_methods
import matplotlib.pyplot as plt
from triangulation.triangulation_toolkit import init_kalman_param, update_kalman
import triangulation.magneto_imu_toolkit as mag_imu
from triangulation.vid_toolkit import angle_between_vectors
from types import SimpleNamespace
import prettytable as pt


sampling_rate = 100
T = 1 / sampling_rate

init_params = {'quat_init': [0, 0, 0.707, 0.707], 'acc_noise': 150 * 1e-6 * GRAVITY * np.sqrt(sampling_rate),
               'R_acc_diff_coeff': 20, 'gyro_noise': np.deg2rad(0.014 * np.sqrt(sampling_rate)), 'vid_noise': 2e4,
               'R_mag_diff_coeff': 0, 'mag_noise': 10}


for subject in SUBJECTS[0:6]:
    print('\n' + subject)
    params_shank_static, _ = init_kalman_param_static(subject, 'SHANK')
    params_thigh_static, knee_angles_vicon_static = init_kalman_param_static(subject, 'THIGH')      # confidence of right camera for AP and V axis; back camera for ML and V axis
    R_shank_body_sens = np.eye(3) @ params_shank_static.R_glob_sens_static        # this is correct
    R_thigh_body_sens = np.eye(3) @ params_thigh_static.R_glob_sens_static
    tb = pt.PrettyTable()
    tb.field_names = ['Trial'] + [axis + ' - ' + method for axis in ['FE', 'AA'] for method in ['Vid_IMU', 'Magn_IMU', 'Video']]

    for trial in TRIALS[:]:            # TRIALS STATIC_TRIALS
        """ vid-IMU """
        params_shank, _, _ = init_kalman_param(subject, trial, 'SHANK', SimpleNamespace(**init_params))
        params_thigh, trial_data, knee_angles_vicon = init_kalman_param(subject, trial, 'THIGH', SimpleNamespace(**init_params))
        for k in range(1, trial_data.shape[0]-1):
            update_kalman(params_thigh, params_thigh_static, T, k)
            update_kalman(params_shank, params_shank_static, T, k)
        knee_angles_vid_imu_esti = q_to_knee_angle(params_shank.q_esti, params_thigh.q_esti, R_shank_body_sens, R_thigh_body_sens)
        knee_angles_vicon = knee_angles_vicon - np.mean(knee_angles_vicon_static, axis=0)     # to remove static knee angle

        """ magneto-IMU """
        params_magneto_shank, _, _ = mag_imu.init_kalman_param(subject, trial, 'SHANK', SimpleNamespace(**init_params))
        params_magneto_thigh, _, _= mag_imu.init_kalman_param(subject, trial, 'THIGH', SimpleNamespace(**init_params))
        for k in range(1, trial_data.shape[0]-1):
            mag_imu.update_kalman(params_magneto_thigh, params_thigh_static, T, k)
            mag_imu.update_kalman(params_magneto_shank, params_shank_static, T, k)
        knee_angles_mag_imu_esti = q_to_knee_angle(params_magneto_shank.q_esti, params_magneto_thigh.q_esti, R_shank_body_sens, R_thigh_body_sens)

        """ vid only """
        knee_angles_vid_esti, _, _ = angle_between_vectors(subject, trial)

        """ Compare results"""
        compare_three_methods(knee_angles_vicon[:, :2], knee_angles_vid_imu_esti[:, :2], knee_angles_mag_imu_esti[:, :2],
                              knee_angles_vid_esti[:, :2], ['Flexion', 'Adduction'], tb, trial)
    print(tb)
    plt.show()


""" Notes """
# 1. subject 1, trial 2, the video-vicon data sync is bad
# !!! apply the V3D filtering to Vid_IMU data !!!

