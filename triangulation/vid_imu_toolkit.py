import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from numpy import cos, sin, arctan2
import os
from triangulation.triangulation_toolkit import compare_axes_results
from const import DATA_PATH, GRAVITY, TARGETS_LIST
from sklearn.metrics import mean_squared_error as mse
from types import SimpleNamespace
from transforms3d.quaternions import rotate_vector, mat2quat, quat2mat
from transforms3d.euler import mat2euler


def compare_three_methods(vicon_data, vid_imu, mag_imu, vid_only, axes, tb, trial, start=5000, end=6500):
    mses = [round(np.sqrt(mse(vicon_data[:, i_axis], esti_data[:, i_axis])), 2)
            for i_axis in range(len(axes)) for esti_data in [vid_imu, mag_imu, vid_only]]
    tb.add_row([trial] + mses)
    for i_axis, axis in enumerate(axes):
        plt.figure(figsize=(8, 5))
        plt.title(axis)
        plt.ylabel('angle (°)')
        plt.xlabel('sample')
        plt.plot(vicon_data[start:end, i_axis], label='OMC')
        plt.plot(vid_imu[start:end, i_axis], '--', label='Video-IMU')
        plt.plot(mag_imu[start:end, i_axis], '--', label='Magneto-IMU')
        plt.plot(vid_only[start:end, i_axis], '--', label='Video only')
        plt.legend()
        plt.grid()


def calibrate_segment_in_global_frame(segment_in_glob_static, R_sens_glob):
    """ Note that segment_in_sens is constant after sensor placement. """
    segment_in_sens = R_sens_glob @ segment_in_glob_static
    return segment_in_sens / np.linalg.norm(segment_in_sens)        # this is a const


def calibrate_segment_in_sensor_frame(acc_static):
    ax, ay, az = np.mean(acc_static, axis=0)
    r = arctan2(ax, ay)
    p = arctan2(-az, cos(r)*ay + sin(r)*ax)
    print('IMU sensor calibration, roll = {:5.2f} deg, pitch = {:5.2f} deg'.
          format(np.rad2deg(r), np.rad2deg(p)))     # to check, r and p should be small
    Cr, Sr, Cp, Sp = sym.symbols('Cr, Sr, Cp, Sp', constant=True)
    R_glob_sens = print_orientation_cali_mat()
    R_glob_sens = np.array(R_glob_sens.subs({Cr: cos(r), Sr: sin(r), Cp: cos(p), Sp: sin(p)})).astype(np.float64)
    return R_glob_sens


def print_orientation_cali_mat():
    Cr, Sr, Cp, Sp, ax, ay, az = sym.symbols('Cr, Sr, Cp, Sp, ax, ay, az', constant=True)
    R_coordinate_switch = sym.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    R_pitch = sym.Matrix([[1, 0, 0], [0, Cp, -Sp], [0, Sp, Cp]])
    R_roll = sym.Matrix([[Cr, -Sr, 0], [Sr, Cr, 0], [0, 0, 1]])
    R_glob_sens = R_coordinate_switch * R_pitch * R_roll

    # exp = R_glob_sens * sym.Matrix([ax, ay, az])
    # collected = sym.expand(exp[1])
    # collected = sym.collect(collected, [Cp, Sp])
    # print('For IMU calibration,', end=' ')
    # print(collected)        # assist my derivation, only need once
    return R_glob_sens


def print_h_mat():
    qk0, qk1, qk2, qk3, sx, sy, sz = sym.symbols('qk0, qk1, qk2, qk3, sx, sy, sz', constant=True)
    R_sens_glob = sym.Matrix([[qk0**2 + qk1**2 - qk2**2 - qk3**2, 2*qk1*qk2+2*qk0*qk3, 2*qk1*qk3-2*qk0*qk2],
                              [2*qk1*qk2-2*qk0*qk3, qk0**2 - qk1**2 + qk2**2 - qk3**2, 2*qk2*qk3+2*qk0*qk1],
                              [2*qk1*qk3+2*qk0*qk2, 2*qk2*qk3-2*qk0*qk1, qk0**2 - qk1**2 - qk2**2 + qk3**2]])
    R_glob_sens = R_sens_glob.T
    segment_in_sens = sym.Matrix([sx, sy, sz])
    segment_in_glob = R_glob_sens * segment_in_sens
    H_k = segment_in_glob.jacobian([qk0, qk1, qk2, qk3])
    print('For IMU H_k_vid and h_vid')
    print(segment_in_glob.T)
    print(H_k)


def get_vicon_orientation(data_df, segment):
        if segment == 'R_SHANK':
            knee_l, knee_r = data_df[['RFME_X', 'RFME_Y', 'RFME_Z']].values, data_df[['RFLE_X', 'RFLE_Y', 'RFLE_Z']].values
            ankle_l, ankle_r = data_df[['RTAM_X', 'RTAM_Y', 'RTAM_Z']].values, data_df[
                ['RFAL_X', 'RFAL_Y', 'RFAL_Z']].values
            shank_ml = (knee_l - knee_r) / 2
            segment_y = ((knee_l + knee_r) - (ankle_l + ankle_r)) / 2
            segment_z = np.cross(shank_ml, segment_y)
            segment_x = np.cross(segment_y, segment_z)
        elif segment == 'R_THIGH':
            knee_l, knee_r = data_df[['RFME_X', 'RFME_Y', 'RFME_Z']].values, data_df[['RFLE_X', 'RFLE_Y', 'RFLE_Z']].values
            hip = data_df[['RFT_X', 'RFT_Y', 'RFT_Z']].values
            shank_ml = (knee_l - knee_r) / 2
            segment_y = hip - (knee_l + knee_r) / 2
            segment_z = np.cross(shank_ml, segment_y)
            segment_x = np.cross(segment_y, segment_z)
        else:
            raise ValueError('Incorrect segment value')

        fun_norm_vect = lambda v: v / np.linalg.norm(v)
        segment_x = np.apply_along_axis(fun_norm_vect, 1, segment_x)
        segment_y = np.apply_along_axis(fun_norm_vect, 1, segment_y)
        segment_z = np.apply_along_axis(fun_norm_vect, 1, segment_z)

        R_body_glob = np.array([segment_x, segment_y, segment_z])
        R_body_glob = np.swapaxes(R_body_glob, 0, 1)
        R_glob_body = np.swapaxes(R_body_glob, 1, 2)

        def temp_fun(R):
            if np.isnan(R).any():
                return np.array([1, 0, 0, 0])
            else:
                quat = mat2quat(R)
                if quat[3] < 0:
                    quat = - quat
                return quat / np.linalg.norm(quat)

        q_vicon = np.array(list(map(temp_fun, R_glob_body)))

        # for i_axis in range(4):
        #     plt.figure()
        #     plt.plot(q_vicon[:, i_axis])
        #     plt.plot(data_df['Quat' + str(i_axis+1) + '_R_SHANK'])
        # plt.show()

        return q_vicon, segment_x, segment_y, segment_z


def init_kalman_param_static(subject, segment):
    if segment == 'SHANK':
        upper_joint, lower_joint = 'RKnee', 'RAnkle'
    elif segment == 'THIGH':
        upper_joint, lower_joint = 'RHip', 'RKnee'
    else:
        raise ValueError('Incorrect segment name')
    params_static = SimpleNamespace()
    trial_static_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', 'static_back.csv'), index_col=0)
    knee_angles_vicon_static = trial_static_data[TARGETS_LIST[:3]].values

    vid_data_static = pd.read_csv(os.path.join(DATA_PATH, subject, 'triangulated', 'static_back.csv'), index_col=0)
    upper_joint_col = [upper_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
    lower_joint_col = [lower_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
    segment_in_glob_static = np.mean(vid_data_static[upper_joint_col].values - vid_data_static[lower_joint_col].values, axis=0)

    params_static.R_glob_sens_static = calibrate_segment_in_sensor_frame(
        trial_static_data[['AccelX_R_'+segment, 'AccelY_R_'+segment, 'AccelZ_R_'+segment]].values)
    params_static.segment_length = norm(segment_in_glob_static)
    R_sens_glob_static = params_static.R_glob_sens_static.T
    params_static.segment_in_sens = calibrate_segment_in_global_frame(segment_in_glob_static, R_sens_glob_static)
    print('{} vector in sensor frame: {}'.format(segment, params_static.segment_in_sens))
    return params_static, knee_angles_vicon_static


def init_kalman_param(subject, trial, segment, init_params):
    trial_data_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\'\
                     + subject + '\combined\\' + trial + '.csv'
    trial_data = pd.read_csv(trial_data_dir, index_col=False)
    knee_angles_vicon = trial_data[TARGETS_LIST[:3]].values
    data_len = trial_data.shape[0]

    if segment == 'SHANK':
        upper_joint, lower_joint = 'RKnee', 'RAnkle'
    elif segment == 'THIGH':
        upper_joint, lower_joint = 'RHip', 'RKnee'
    else:
        raise ValueError('Incorrect segment name')

    params = SimpleNamespace()
    params.q_esti = np.zeros([data_len, 4])
    params.q_esti[0, :] = init_params.quat_init
    params.P = np.zeros([4, 4, data_len])
    params.P[:, :, 0] = np.eye(4)
    params.Q = init_params.gyro_noise * np.eye(4)
    params.acc = trial_data[['AccelX_R_'+segment, 'AccelY_R_'+segment, 'AccelZ_R_'+segment]].values
    params.gyr = np.deg2rad(trial_data[['GyroX_R_'+segment, 'GyroY_R_'+segment, 'GyroZ_R_'+segment]].values)

    vid_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'triangulated', trial+'.csv'), index_col=0)
    upper_joint_col = [upper_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
    lower_joint_col = [lower_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
    params.segment_in_glob = vid_data[upper_joint_col].values - vid_data[lower_joint_col].values
    params.segment_confidence = trial_data[upper_joint+'_probability_90'] * trial_data[upper_joint+'_probability_180'] * \
                                trial_data[lower_joint+'_probability_90'] * trial_data[lower_joint+'_probability_180']

    params.R_acc_diff_coeff = init_params.R_acc_diff_coeff
    params.R_acc_base = np.eye(3) * init_params.acc_noise
    params.R_vid_base = np.eye(3) * init_params.vid_noise
    params.V_acc = np.eye(3)               # correlation between the acc noise and angular position
    params.V_vid = np.eye(3)
    return params, trial_data, knee_angles_vicon


def update_kalman(params, params_static, T, k):
    gyr, q_esti, acc, P, Q, R_acc_base, R_acc_diff_coeff, V_acc, R_vid_base, V_vid, segment_confidence, segment_in_glob, segment_length = \
        params.gyr, params.q_esti, params.acc, params.P, params.Q, params.R_acc_base, params.R_acc_diff_coeff,\
        params.V_acc, params.R_vid_base, params.V_vid, params.segment_confidence, params.segment_in_glob, params_static.segment_length
    sx, sy, sz = params_static.segment_in_sens

    """ a prior system estimation """
    omega = np.array([[0, -gyr[k, 0], -gyr[k, 1], -gyr[k, 2]],
                      [gyr[k, 0], 0, gyr[k, 2], -gyr[k, 1]],
                      [gyr[k, 1], -gyr[k, 2], 0, gyr[k, 0]],
                      [gyr[k, 2], gyr[k, 1], -gyr[k, 0], 0]])
    Ak = np.eye(4) + 0.5 * omega * T
    qk_ = Ak @ q_esti[k-1]
    qk_ = qk_ / norm(qk_)      # should there be a norm? !!!
    qk0, qk1, qk2, qk3 = qk_

    P_k_minus = Ak @ P[:, :, k-1] @ Ak.T + Q

    """ correction stage 1, based on acc. Note that the h_vid, z_vid, and R are normalized by the gravity """
    H_k_acc = 2 * np.array([[-qk2, qk3, -qk0, qk1],
                            [qk1, qk0, qk3, qk2],
                            [qk0, -qk1, -qk2, qk3]])
    acc_diff = norm(rotate_vector(acc[k], qk_) - np.array([0, 0, GRAVITY]))
    R_acc = R_acc_base + R_acc_diff_coeff * np.array([
        [acc_diff**2, 0, 0],
        [0, acc_diff**2, 0],
        [0, 0, acc_diff**2]])
    R_acc = R_acc / GRAVITY
    K_k_acc = P_k_minus @ H_k_acc.T @ np.matrix(H_k_acc @ P_k_minus @ H_k_acc.T + V_acc @ R_acc @ V_acc.T).I

    h_acc = np.array([2*qk1*qk3 - 2*qk0*qk2,
                      2*qk0*qk1 + 2*qk2*qk3,
                      qk0**2 - qk1**2 - qk2**2 + qk3**2])
    z_acc = acc[k].T / norm(acc[k])
    q_acc_eps = K_k_acc @ (z_acc - h_acc)

    """ correction stage 2, based on vid. Note that the h_vid, z_vid, and R are normalized by the segment length """
    H_k_vid = np.array([[2*qk0*sx + 2*qk2*sz - 2*qk3*sy, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz, 2*qk0*sz + 2*qk1*sy - 2*qk2*sx, -2*qk0*sy + 2*qk1*sz - 2*qk3*sx],
                        [2*qk0*sy - 2*qk1*sz + 2*qk3*sx, -2*qk0*sz - 2*qk1*sy + 2*qk2*sx, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz, 2*qk0*sx + 2*qk2*sz - 2*qk3*sy],
                        [2*qk0*sz + 2*qk1*sy - 2*qk2*sx, 2*qk0*sy - 2*qk1*sz + 2*qk3*sx, -2*qk0*sx - 2*qk2*sz + 2*qk3*sy, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz]])
    R_vid = R_vid_base / segment_confidence[k] / segment_length     # this could be more innovative
    K_k_vid = P_k_minus @ H_k_vid.T @ np.matrix(H_k_vid @ P_k_minus @ H_k_vid.T + V_vid @ R_vid @ V_vid.T).I

    h_vid = np.array([sx*(qk0**2 + qk1**2 - qk2**2 - qk3**2) + sy*(-2*qk0*qk3 + 2*qk1*qk2) + sz*(2*qk0*qk2 + 2*qk1*qk3),
                      sx*(2*qk0*qk3 + 2*qk1*qk2) + sy*(qk0**2 - qk1**2 + qk2**2 - qk3**2) + sz*(-2*qk0*qk1 + 2*qk2*qk3),
                      sx*(-2*qk0*qk2 + 2*qk1*qk3) + sy*(2*qk0*qk1 + 2*qk2*qk3) + sz*(qk0**2 - qk1**2 - qk2**2 + qk3**2)])
    z_vid = segment_in_glob[k].T / norm(segment_in_glob[k])
    q_vid_eps = K_k_vid @ (z_vid - h_vid)

    """ combine """
    qk = qk_ + q_acc_eps + q_vid_eps
    q_esti[k] = qk / norm(qk)
    P[:, :, k] = (np.eye(4) - K_k_vid @ H_k_vid) @ (np.eye(4) - K_k_acc @ H_k_acc) @ P_k_minus


def q_to_knee_angle(q_shank_glob_sens, q_thigh_glob_sens, R_shank_body_sens, R_thigh_body_sens):
    data_len = q_shank_glob_sens.shape[0]
    R_shankbody_thighbody = np.zeros([data_len, 3, 3])
    knee_angles = np.zeros([data_len, 3])
    for k in range(data_len):
        R_shank_glob_body = quat2mat(q_shank_glob_sens[k]) @ R_shank_body_sens.T
        R_thigh_glob_body = quat2mat(q_thigh_glob_sens[k]) @ R_thigh_body_sens.T
        R_shankbody_thighbody[k] = R_shank_glob_body.T @ R_thigh_glob_body
        knee_angles[k] = mat2euler(R_shankbody_thighbody[k]) * np.array([1, -1, 1])

    return np.rad2deg(knee_angles)


def figure_for_FE_AA_angles(vicon_data, esti_data, axes=['X', 'Y', 'Z'], start=0, end=1000, title=''):
    print('{:10}'.format(title), end='')
    [print('{:5.1f}\t'.format(np.sqrt(mse(vicon_data[:, i_axis], esti_data[:, i_axis]))), end='')
     for i_axis, axis in enumerate(axes)]
    print()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i_axis, axis in enumerate(axes):      # 'Medio-lateral', 'Anterior-posterior', 'Vertical'
        plt.figure(figsize=(6, 3.5))
        line, = plt.plot(vicon_data[start:end, i_axis], label=axis + ' - Mocap', color=colors[i_axis])
        plt.plot(esti_data[start:end, i_axis], '--', color=line.get_color(), label=axis + ' - Smartphone')
        plt.legend()
        plt.grid()
        plt.title(title)
        plt.ylabel(axis + ' angle (°)')
        plt.xlabel('sample')
        plt.tight_layout()


def plot_q_for_debug(trial_data, q_shank_esti, q_thigh_esti):
    q_vicon_shank, shank_x, shank_y, shank_z = get_vicon_orientation(trial_data, 'R_SHANK')
    q_vicon_thigh, thigh_x, thigh_y, thigh_z = get_vicon_orientation(trial_data, 'R_THIGH')

    # acc_global_esti, acc_global_vicon = np.zeros(params_thigh.acc.shape), np.zeros(params_thigh.acc.shape)
    # for i in range(1, trial_data.shape[0]-1):
    #     acc_global_esti[i] = rotate_vector(params_thigh.acc[i], params_thigh.q_esti[i])
    #     acc_global_vicon[i] = rotate_vector(params_thigh.acc[i], q_vicon_shank[i])
    # for i_axis in range(3):
    #     plt.figure()
    #     plt.plot(acc_global_esti[:, i_axis])
    #     plt.plot(acc_global_vicon[:, i_axis])

    compare_axes_results(q_vicon_shank, q_shank_esti, ['q0', 'q1', 'q2', 'q3'], title='shank orientation', end=shank_x.shape[0])
    compare_axes_results(q_vicon_thigh, q_thigh_esti, ['q0', 'q1', 'q2', 'q3'], title='thigh orientation', end=thigh_x.shape[0])
