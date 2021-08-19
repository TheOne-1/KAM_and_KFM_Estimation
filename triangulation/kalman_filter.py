import numpy as np
from const import SUBJECTS, GRAVITY, TRIALS, STATIC_TRIALS
from triangulation.triangulation_toolkit import init_kalman_param_static, update_kalman, q_to_knee_angle, \
    compare_axes_results, get_vicon_orientation, init_kalman_param
import matplotlib.pyplot as plt
from types import SimpleNamespace
from transforms3d.quaternions import rotate_vector, mat2quat


# during static trial, all three angles = 0, use it to calibrate !!! very important, potentially solving discrepancy between Openpose and markers
subject = SUBJECTS[0]
trial = TRIALS[0]            # TRIALS STATIC_TRIALS
sampling_rate = 100
T = 1 / sampling_rate

init_params = {'quat_init': [0, 0, 0.707, 0.707], 'acc_noise': 150 * 1e-6 * GRAVITY * np.sqrt(sampling_rate),
               'gyro_noise': np.deg2rad(0.014 * np.sqrt(sampling_rate)), 'vid_noise': 1e5}

params_shank_static = init_kalman_param_static(subject, 'SHANK')
params_shank, _, _ = init_kalman_param(subject, trial, 'SHANK', SimpleNamespace(**init_params))
params_thigh_static = init_kalman_param_static(subject, 'THIGH')
params_thigh, trial_data, knee_angles_vicon = init_kalman_param(subject, trial, 'THIGH', SimpleNamespace(**init_params))

for k in range(1, trial_data.shape[0]-1):
    update_kalman(params_thigh, params_thigh_static, T, k)
    update_kalman(params_shank, params_shank_static, T, k)



""" for debug """
# q_vicon_shank, shank_x, shank_y, shank_z = get_vicon_orientation(trial_data, 'R_SHANK')
# q_vicon_thigh, thigh_x, thigh_y, thigh_z = get_vicon_orientation(trial_data, 'R_THIGH')
#
# acc_global_esti, acc_global_vicon = np.zeros(params_thigh.acc.shape), np.zeros(params_thigh.acc.shape)
# for i in range(1, trial_data.shape[0]-1):
#     acc_global_esti[i] = rotate_vector(params_thigh.acc[i], params_thigh.q_esti[i])
#     acc_global_vicon[i] = rotate_vector(params_thigh.acc[i], q_vicon_shank[i])
# for i_axis in range(3):
#     plt.figure()
#     plt.plot(acc_global_esti[:, i_axis])
#     plt.plot(acc_global_vicon[:, i_axis])

# compare_axes_results(q_vicon_shank, params_shank.q_esti, ['q0', 'q1', 'q2', 'q3'], title='shank orientation')
# compare_axes_results(q_vicon_thigh, params_thigh.q_esti, ['q0', 'q1', 'q2', 'q3'], title='thigh orientation')

R_shank_body_sens = np.eye(3) @ params_shank_static.R_glob_sens_static        # this is correct
R_thigh_body_sens = np.eye(3) @ params_thigh_static.R_glob_sens_static
knee_angles_esti = q_to_knee_angle(params_shank.q_esti, params_thigh.q_esti, R_shank_body_sens, R_thigh_body_sens)

compare_axes_results(knee_angles_vicon, knee_angles_esti, ['FE', 'AA', 'IE'])       # , start=0, end=trial_data.shape[0]










plt.show()


