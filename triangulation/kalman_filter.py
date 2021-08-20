import numpy as np
from const import SUBJECTS, GRAVITY, TRIALS, STATIC_TRIALS
from triangulation.triangulation_toolkit import init_kalman_param_static, update_kalman, q_to_knee_angle, \
    compare_axes_results, get_vicon_orientation, init_kalman_param, get_knee_angle_vicon_from_raw_marker, \
    figure_for_FE_AA_angles
import matplotlib.pyplot as plt
from types import SimpleNamespace
from transforms3d.quaternions import rotate_vector, mat2quat


# during static trial, all three angles = 0, use it to calibrate !!! very important, potentially solving discrepancy between Openpose and markers
sampling_rate = 100
T = 1 / sampling_rate

init_params = {'quat_init': [0, 0, 0.707, 0.707], 'acc_noise': 150 * 1e-6 * GRAVITY * np.sqrt(sampling_rate),
               'R_acc_diff_coeff': 20, 'gyro_noise': np.deg2rad(0.014 * np.sqrt(sampling_rate)), 'vid_noise': 1e5}

for subject in SUBJECTS[0:2]:
    print('\n' + subject)
    params_shank_static, _ = init_kalman_param_static(subject, 'SHANK')
    params_thigh_static, knee_angles_vicon_static = init_kalman_param_static(subject, 'THIGH')      # confidence of right camera for AP and V axis; back camera for ML and V axis
    R_shank_body_sens = np.eye(3) @ params_shank_static.R_glob_sens_static        # this is correct
    R_thigh_body_sens = np.eye(3) @ params_thigh_static.R_glob_sens_static
    print('{:10} FE \tAA \tIE'.format(''))
    for trial in TRIALS:            # TRIALS STATIC_TRIALS
        params_shank, _, _ = init_kalman_param(subject, trial, 'SHANK', SimpleNamespace(**init_params))
        params_thigh, trial_data, knee_angles_vicon = init_kalman_param(subject, trial, 'THIGH', SimpleNamespace(**init_params))

        for k in range(1, trial_data.shape[0]-1):
            update_kalman(params_thigh, params_thigh_static, T, k)
            update_kalman(params_shank, params_shank_static, T, k)

        knee_angles_esti = q_to_knee_angle(params_shank.q_esti, params_thigh.q_esti, R_shank_body_sens, R_thigh_body_sens)
        knee_angles_vicon = knee_angles_vicon - np.mean(knee_angles_vicon_static, axis=0)     # to remove static knee angle
        knee_angle_vicon = get_knee_angle_vicon_from_raw_marker(trial_data)
        figure_for_FE_AA_angles(knee_angles_vicon, knee_angles_esti, ['Flexion', 'Adduction'],
                                start=1000, end=1400)


    plt.show()


