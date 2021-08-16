import numpy as np
import pandas as pd
from const import SUBJECTS, camera_pairs_all_sub_90, camera_pairs_all_sub_180
from triangulation.triangulation_toolkit import calibrate_leg_in_global_frame, calibrate_segment_in_sensor_frame, \
    triangulate, print_h_mat, compare_axes_results, get_vicon_orientation
import matplotlib.pyplot as plt
from transforms3d.quaternions import qinverse, rotate_vector, quat2axangle
from transforms3d.euler import quat2euler


subject = SUBJECTS[0]
sampling_rate = 100
quat_init = [0.707, -0.707, 0, 0]
gyroscope_noise = 1         # check datasheet !!!

trial_data_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\' + subject + '\combined\\baseline.csv'
trial_data = pd.read_csv(trial_data_dir, index_col=False)
trial_static_data_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\' + subject + '\combined\\static_back.csv'
trial_static_data = pd.read_csv(trial_static_data_dir, index_col=False)
data_len = trial_data.shape[0]
T = 1 / sampling_rate
q_esti = np.zeros([data_len, 4])
q_esti[0, :] = quat_init
P = np.zeros([4, 4, data_len])
P[:, :, 0] = np.eye(4)
Q = 1e-4 * np.eye(4)        # gyro noise
R_acc = 0.5 * np.eye(3)         # noise of gravity measurement !!! make it related to acc magnitude
V_acc = np.eye(3)               # correlation between the acc noise and angular position

R_vid = 0.5 * np.eye(3)     # !!! make it related to estimation confidence
V_vid = np.eye(3)

acc = trial_data[['AccelX_R_SHANK', 'AccelY_R_SHANK', 'AccelZ_R_SHANK']].values
gyr = np.deg2rad(trial_data[['GyroX_R_SHANK', 'GyroY_R_SHANK', 'AccelZ_R_SHANK']].values)
q_sage = trial_data[['Quat1_R_SHANK', 'Quat2_R_SHANK', 'Quat3_R_SHANK', 'Quat4_R_SHANK']].values

vid_data = triangulate(['RKnee', 'RAnkle'], trial_data, camera_pairs_all_sub_90[subject], camera_pairs_all_sub_180[subject])
shank_in_glob = vid_data['RKnee'] - vid_data['RAnkle']     # !!! shank_in_n

# confirmed that shank_in_b_static is correct
vid_data_static = triangulate(['RKnee', 'RAnkle'], trial_static_data, camera_pairs_all_sub_90[subject], camera_pairs_all_sub_180[subject])
R_sens_to_glob_static = calibrate_segment_in_sensor_frame(trial_static_data[['AccelX_R_SHANK', 'AccelY_R_SHANK', 'AccelZ_R_SHANK']].values)
sx, sy, sz = calibrate_leg_in_global_frame(np.mean(vid_data_static['RKnee'], axis=0), np.mean(vid_data_static['RAnkle'], axis=0), R_sens_to_glob_static.T)
print('Shank vector in sensor frame: {}, {}, {}'.format(sx, sy, sz))
# print_h_mat()

for k in range(1, data_len-1):
    """ a prior system estimation """
    omega = np.array([[0, -gyr[k, 0], -gyr[k, 1], -gyr[k, 2]],
                      [gyr[k, 0], 0, gyr[k, 2], -gyr[k, 1]],
                      [gyr[k, 1], -gyr[k, 2], 0, gyr[k, 0]],
                      [gyr[k, 2], gyr[k, 1], -gyr[k, 0], 0]])
    Ak = np.eye(4) + 0.5 * omega * T
    qk_ = Ak @ q_esti[k-1]
    qk0, qk1, qk2, qk3 = qk_

    P_k_minus = Ak @ P[:, :, k-1] @ Ak.T + Q

    """ correction stage 1, based on acc """
    H_k_acc = 2 * np.array([[-qk2, qk3, -qk0, qk1],
                            [qk1, qk0, qk3, qk2],
                            [qk0, -qk1, -qk2, qk3]])
    K_k_acc = P_k_minus @ H_k_acc.T @ np.matrix(H_k_acc @ P_k_minus @ H_k_acc.T + V_acc @ R_acc @ V_acc.T).I

    h_acc = np.array([2*qk1*qk3 - 2*qk0*qk2,
                      2*qk0*qk1 + 2*qk2*qk3,
                      qk0**2 - qk1**2 - qk2**2 + qk3**2])
    z_acc = acc[k, :].T / np.linalg.norm(acc[k, :])
    q_acc_eps = K_k_acc @ (z_acc - h_acc)

    """ correction stage 2, based on vid """
    H_k_vid = 2 * np.array([[2*qk0*sx - 2*qk2*sz + 2*qk3*sy, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz, -2*qk0*sz + 2*qk1*sy - 2*qk2*sx, 2*qk0*sy + 2*qk1*sz - 2*qk3*sx],
                            [2*qk0*sy + 2*qk1*sz - 2*qk3*sx, 2*qk0*sz - 2*qk1*sy + 2*qk2*sx, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz, -2*qk0*sx + 2*qk2*sz - 2*qk3*sy],
                            [2*qk0*sz - 2*qk1*sy + 2*qk2*sx, -2*qk0*sy - 2*qk1*sz + 2*qk3*sx, 2*qk0*sx - 2*qk2*sz + 2*qk3*sy, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz]])
    K_k_vid = P_k_minus @ H_k_vid.T @ np.matrix(H_k_vid @ P_k_minus @ H_k_vid.T + V_vid @ R_vid @ V_vid.T).I

    h_vid = np.array([sx*(qk0**2 + qk1**2 - qk2**2 - qk3**2) + sy*(2*qk0*qk3 + 2*qk1*qk2) + sz*(-2*qk0*qk2 + 2*qk1*qk3),
                      sx*(-2*qk0*qk3 + 2*qk1*qk2) + sy*(qk0**2 - qk1**2 + qk2**2 - qk3**2) + sz*(2*qk0*qk1 + 2*qk2*qk3),
                      sx*(2*qk0*qk2 + 2*qk1*qk3) + sy*(-2*qk0*qk1 + 2*qk2*qk3) + sz*(qk0**2 - qk1**2 - qk2**2 + qk3**2)])
    z_vid = shank_in_glob[k, :].T / np.linalg.norm(shank_in_glob[k, :])
    q_vid_eps = K_k_vid @ (z_vid - h_vid)

    """ combine """
    qk = qk_ + q_acc_eps + q_vid_eps
    q_esti[k, :] = qk / np.linalg.norm(qk)
    P[:, :, k] = (np.eye(4) - K_k_vid @ H_k_vid) @ (np.eye(4) - K_k_acc @ H_k_acc) @ P_k_minus



q_vicon_shank, shank_x, shank_y, shank_z = get_vicon_orientation(trial_data, 'R_SHANK')

acc_global_esti, acc_global_vicon = np.zeros(acc.shape), np.zeros(acc.shape)
for i in range(1, data_len):
    acc_global_esti[i] = rotate_vector(shank_in_glob[i], qinverse(q_esti[i]))
    acc_global_vicon[i] = rotate_vector(shank_in_glob[i], qinverse(q_vicon_shank[i]))

for i_axis in range(3):
    plt.figure()
    plt.plot(acc_global_esti[:, i_axis])
    plt.plot(acc_global_vicon[:, i_axis])
plt.show()



angle_vicon, angle_esti = np.zeros([data_len, 3]), np.zeros([data_len, 3])
for i in range(1, data_len):
    angle_vicon[i], angle_esti[i] = quat2euler(q_vicon_shank[i]), quat2euler(q_esti[i])
compare_axes_results(angle_vicon, angle_esti, ['x', 'y', 'z'])
# compare_axes_results(q_vicon_shank, q_esti, ['q0', 'q1', 'q2', 'q3'])
plt.show()


