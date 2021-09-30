import numpy as np
from const import SUBJECTS, GRAVITY, TRIALS
from triangulation.vid_imu_toolkit import KalmanFilterVidIMU, KalmanFilterMagIMU, VidOnlyKneeAngle, q_to_knee_angle, \
    MadgwickVidIMU, MadgwickMagIMU, compare_three_methods, compare_axes_results, plot_q_for_debug, plot_euler_angle_for_debug
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

angles_to_check = ['FE', 'AA', 'IE']

for subject in SUBJECTS[3:6]:
    print('\n' + subject)
    tb = pt.PrettyTable()
    tb.field_names = ['Trial'] + [axis + ' - ' + method for axis in angles_to_check for method in ['Vid_IMU', 'Mag_IMU', 'Video']]

    for trial in TRIALS[0:1]:            # TRIALS STATIC_TRIALS
        """ vid-IMU """
        shank_vid_imu = MadgwickVidIMU(subject, 'SHANK', trial, SimpleNamespace(**init_params_vid_imu))
        thigh_vid_imu = MadgwickVidIMU(subject, 'THIGH', trial, SimpleNamespace(**init_params_vid_imu))
        for k in range(1, shank_vid_imu.trial_data.shape[0]):
            shank_vid_imu.update(k)
            thigh_vid_imu.update(k)
        R_shank_body_sens, R_thigh_body_sens = shank_vid_imu.R_body_sens, thigh_vid_imu.R_body_sens
        knee_angles_vid_imu_esti = q_to_knee_angle(shank_vid_imu.params.q_esti, thigh_vid_imu.params.q_esti,
                                                   R_shank_body_sens, R_thigh_body_sens)
        knee_angles_vicon = shank_vid_imu.knee_angles_vicon - np.mean(shank_vid_imu.knee_angles_vicon_static, axis=0)
        # plot_euler_angle_for_debug(shank_vid_imu.trial_data, shank_vid_imu.params.q_esti, thigh_vid_imu.params.q_esti)
        # plt.show()

        """ magneto-IMU """
        shank_mag_imu = MadgwickMagIMU(subject, 'SHANK', trial, SimpleNamespace(**init_params_mag_imu))
        thigh_mag_imu = MadgwickMagIMU(subject, 'THIGH', trial, SimpleNamespace(**init_params_mag_imu))
        for k in range(1, shank_mag_imu.trial_data.shape[0]):
            shank_mag_imu.update(k)
            thigh_mag_imu.update(k)
        knee_angles_mag_imu_esti = q_to_knee_angle(shank_mag_imu.params.q_esti, thigh_mag_imu.params.q_esti,
                                                   R_shank_body_sens, R_thigh_body_sens)
        # plot_q_for_debug(shank_mag_imu.trial_data, shank_mag_imu.params.q_esti, thigh_mag_imu.params.q_esti)
        # plt.show()

        """ vid only """
        knee_angles_vid_esti, _, _ = VidOnlyKneeAngle.angle_between_vectors(subject, trial)

        """ Compare results"""
        # knee_angles_vid_imu_esti = data_filter(knee_angles_vid_imu_esti, 15, 100, 2)
        # knee_angles_mag_imu_esti = data_filter(knee_angles_mag_imu_esti, 15, 100, 2)
        # knee_angles_vid_esti = data_filter(knee_angles_vid_esti, 15, 100, 2)
        compare_three_methods(knee_angles_vicon[:, :], knee_angles_vid_imu_esti[:, :], knee_angles_mag_imu_esti[:, :],
                              knee_angles_vid_esti[:, :], angles_to_check, tb, trial, start=0, end=shank_mag_imu.trial_data.shape[0])
    print(tb)
    plt.show()


""" Notes """
# 1. subject 1, trial 2, the video-vicon data sync is bad
# 2. vid-IMU underestimate during stance phase, while magneto-IMU overestimate at the same phase.
#    This might be because different reference homogenous field

