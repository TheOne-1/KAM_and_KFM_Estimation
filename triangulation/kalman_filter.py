import numpy as np
from const import SUBJECTS, GRAVITY, TRIALS, STATIC_TRIALS
from triangulation.triangulation_toolkit import init_kalman_param_static, q_to_knee_angle, \
    compare_axes_results, plot_q_for_debug, figure_for_FE_AA_angles, print_h_mat
import matplotlib.pyplot as plt
from types import SimpleNamespace


method = 'magneto-IMU'          # 'vid-IMU', 'magneto-IMU', 'vid'

sampling_rate = 100
T = 1 / sampling_rate

init_params = {'quat_init': [0, 0, 0.707, 0.707], 'acc_noise': 150 * 1e-6 * GRAVITY * np.sqrt(sampling_rate),
               'R_acc_diff_coeff': 20, 'gyro_noise': np.deg2rad(0.014 * np.sqrt(sampling_rate)), 'vid_noise': 2e4,
               'R_mag_diff_coeff': 0, 'mag_noise': 10}

if method == 'vid-IMU':
    from triangulation.triangulation_toolkit import init_kalman_param, update_kalman
elif method == 'magneto-IMU':
    from triangulation.magneto_imu_toolkit import init_kalman_param, update_kalman
elif method == 'vid':
    pass

for subject in SUBJECTS[0:2]:
    print('\n' + subject)
    params_shank_static, _ = init_kalman_param_static(subject, 'SHANK')
    params_thigh_static, knee_angles_vicon_static = init_kalman_param_static(subject, 'THIGH')      # confidence of right camera for AP and V axis; back camera for ML and V axis
    R_shank_body_sens = np.eye(3) @ params_shank_static.R_glob_sens_static        # this is correct
    R_thigh_body_sens = np.eye(3) @ params_thigh_static.R_glob_sens_static
    print('{:10} FE \tAA \tIE'.format(''))
    for trial in TRIALS[:1]:            # TRIALS STATIC_TRIALS
        params_shank, _, _ = init_kalman_param(subject, trial, 'SHANK', SimpleNamespace(**init_params))
        params_thigh, trial_data, knee_angles_vicon = init_kalman_param(subject, trial, 'THIGH', SimpleNamespace(**init_params))

        for k in range(1, trial_data.shape[0]-1):
            update_kalman(params_thigh, params_thigh_static, T, k)
            update_kalman(params_shank, params_shank_static, T, k)

        knee_angles_esti = q_to_knee_angle(params_shank.q_esti, params_thigh.q_esti, R_shank_body_sens, R_thigh_body_sens)
        knee_angles_vicon = knee_angles_vicon - np.mean(knee_angles_vicon_static, axis=0)     # to remove static knee angle
        compare_axes_results(knee_angles_vicon, knee_angles_esti, ['Flexion', 'Adduction', 'IE'],
                             start=0, end=trial_data.shape[0])

        plot_q_for_debug(trial_data, params_shank.q_esti, params_thigh.q_esti)

    plt.show()


""" Notes """
# 1. subject 1, trial 2, the video-vicon data sync is bad

