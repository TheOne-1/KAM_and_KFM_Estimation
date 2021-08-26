import numpy as np
from const import SUBJECTS, GRAVITY, TRIALS
from triangulation.vid_imu_toolkit import KalmanFilterVidIMU, KalmanFilterMagIMU, VidOnlyKneeAngle, q_to_knee_angle, \
    plot_q_for_debug, print_h_mat, compare_three_methods, compare_axes_results
from wearable_toolkit import data_filter
import matplotlib.pyplot as plt
from types import SimpleNamespace
import prettytable as pt


sampling_rate = 100
init_params = {'quat_init': [0, 0, 0.707, 0.707], 'acc_noise': 150 * 1e-6 * GRAVITY * np.sqrt(sampling_rate),
               'gyro_noise': np.deg2rad(0.014 * np.sqrt(sampling_rate)), 't': 1/sampling_rate}
init_params_vid_imu = {'R_acc_diff_coeff': 20, 'vid_noise': 2e4}
init_params_vid_imu.update(init_params)
init_params_mag_imu = {'R_acc_diff_coeff': 20, 'R_mag_diff_coeff': 0, 'mag_noise': 10}
init_params_mag_imu.update(init_params)

for subject in SUBJECTS[0:6]:
    print('\n' + subject)
    tb = pt.PrettyTable()
    tb.field_names = ['Trial'] + [axis + ' - ' + method for axis in ['FE', 'AA'] for method in ['Vid_IMU', 'Magn_IMU', 'Video']]

    for trial in TRIALS[:1]:            # TRIALS STATIC_TRIALS
        """ vid-IMU """
        kalman_shank = KalmanFilterVidIMU(subject, 'SHANK', trial, SimpleNamespace(**init_params_vid_imu))
        kalman_thigh = KalmanFilterVidIMU(subject, 'THIGH', trial, SimpleNamespace(**init_params_vid_imu))
        for k in range(1, kalman_shank.trial_data.shape[0]):
            kalman_shank.update_kalman(k)
            kalman_thigh.update_kalman(k)
        R_shank_body_sens, R_thigh_body_sens = kalman_shank.R_body_sens, kalman_thigh.R_body_sens
        knee_angles_vid_imu_esti = q_to_knee_angle(kalman_shank.params.q_esti, kalman_thigh.params.q_esti,
                                                   R_shank_body_sens, R_thigh_body_sens)
        knee_angles_vicon = kalman_shank.knee_angles_vicon - np.mean(kalman_shank.knee_angles_vicon_static, axis=0)     # to remove static knee angle

        """ magneto-IMU """
        kalman_shank_mag = KalmanFilterMagIMU(subject, 'SHANK', trial, SimpleNamespace(**init_params_mag_imu))
        kalman_thigh_mag = KalmanFilterMagIMU(subject, 'THIGH', trial, SimpleNamespace(**init_params_mag_imu))
        for k in range(1, kalman_shank_mag.trial_data.shape[0]):
            kalman_shank_mag.update_kalman(k)
            kalman_thigh_mag.update_kalman(k)
        knee_angles_mag_imu_esti = q_to_knee_angle(kalman_shank_mag.params.q_esti, kalman_thigh_mag.params.q_esti,
                                                   R_shank_body_sens, R_thigh_body_sens)

        """ vid only """
        knee_angles_vid_esti, _, _ = VidOnlyKneeAngle.angle_between_vectors(subject, trial)

        """ Compare results"""
        knee_angles_vid_imu_esti = data_filter(knee_angles_vid_imu_esti, 15, 100, 2)
        knee_angles_mag_imu_esti = data_filter(knee_angles_mag_imu_esti, 15, 100, 2)
        knee_angles_vid_esti = data_filter(knee_angles_vid_esti, 15, 100, 2)
        compare_three_methods(knee_angles_vicon[:, :2], knee_angles_vid_imu_esti[:, :2], knee_angles_mag_imu_esti[:, :2],
                              knee_angles_vid_esti[:, :2], ['Flexion', 'Adduction'], tb, trial)
    print(tb)
plt.show()


""" Notes """
# 1. subject 1, trial 2, the video-vicon data sync is bad
# 2. vid-IMU underestimate during stance phase, while magneto-IMU overestimate at the same phase.
#    This might be because different reference homogenous field
