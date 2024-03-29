import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from numpy import cos, sin, arctan2
import os
from triangulation.triangulation_toolkit import compare_axes_results
from const import DATA_PATH, GRAVITY, TARGETS_LIST, TREADMILL_MAG_FIELD
from sklearn.metrics import mean_squared_error as mse
from types import SimpleNamespace
from transforms3d.quaternions import rotate_vector, mat2quat, quat2mat, qmult, qconjugate
from transforms3d.euler import mat2euler, quat2euler


def compare_three_methods(vicon_data, vid_imu, mag_imu, vid_only, axes, tb, trial, start=5000, end=5500):
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


def calibrate_segment_in_sensor_frame(acc_static, segment):
    ax, ay, az = np.mean(acc_static, axis=0)
    r = arctan2(ax, ay)
    p = arctan2(-az, cos(r)*ay + sin(r)*ax)
    print('{} IMU calibration, roll = {:5.2f} deg, pitch = {:5.2f} deg'.
          format(segment, np.rad2deg(r), np.rad2deg(p)))     # to check, r and p should be small
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
    # for i in range(1, trial_data.shape[0]):
    #     acc_global_esti[i] = rotate_vector(params_thigh.acc[i], params_thigh.q_esti[i])
    #     acc_global_vicon[i] = rotate_vector(params_thigh.acc[i], q_vicon_shank[i])
    # for i_axis in range(3):
    #     plt.figure()
    #     plt.plot(acc_global_esti[:, i_axis])
    #     plt.plot(acc_global_vicon[:, i_axis])

    compare_axes_results(q_vicon_shank, q_shank_esti, ['q0', 'q1', 'q2', 'q3'], title='shank orientation', end=shank_x.shape[0])
    compare_axes_results(q_vicon_thigh, q_thigh_esti, ['q0', 'q1', 'q2', 'q3'], title='thigh orientation', end=thigh_x.shape[0])


def plot_euler_angle_for_debug(trial_data, q_shank_esti, q_thigh_esti):
    def apply_fun(q):
        a = np.rad2deg(quat2euler(q))
        if a[2] < -100:
            a[2] += 360
        return a
    q_vicon_shank, shank_x, shank_y, shank_z = get_vicon_orientation(trial_data, 'R_SHANK')
    q_vicon_thigh, thigh_x, thigh_y, thigh_z = get_vicon_orientation(trial_data, 'R_THIGH')
    a1, a2 = np.apply_along_axis(apply_fun, 1, q_vicon_shank), np.apply_along_axis(apply_fun, 1, q_shank_esti)
    a3, a4 = np.apply_along_axis(apply_fun, 1, q_vicon_thigh), np.apply_along_axis(apply_fun, 1, q_thigh_esti)
    compare_axes_results(a1, a2, ['r', 'p', 'y'], title='shank orientation', end=shank_x.shape[0])
    compare_axes_results(a3, a4, ['r', 'p', 'y'], title='thigh orientation', end=thigh_x.shape[0])


class KalmanFilterVidIMU:
    def __init__(self, subject, segment, trial, init_params):
        self.subject = subject
        self.segment = segment
        self.trial = trial
        self.t = init_params.t
        if segment == 'SHANK':
            self.upper_joint, self.lower_joint = 'RKnee', 'RAnkle'
        elif segment == 'THIGH':
            self.upper_joint, self.lower_joint = 'RHip', 'RKnee'
        else:
            raise ValueError('Incorrect segment name')
        self.params_static, self.knee_angles_vicon_static = self.init_kalman_static_param()
        self.params, self.trial_data, self.knee_angles_vicon = self._init_trial_param(init_params)
        self.R_body_sens = np.eye(3) @ self.params_static.R_glob_sens_static

    @staticmethod
    def print_h_mat():
        qk0, qk1, qk2, qk3, sx, sy, sz = sym.symbols('qk0, qk1, qk2, qk3, sx, sy, sz', constant=True)
        R_sens_glob = sym.Matrix(
            [[qk0 ** 2 + qk1 ** 2 - qk2 ** 2 - qk3 ** 2, 2 * qk1 * qk2 + 2 * qk0 * qk3, 2 * qk1 * qk3 - 2 * qk0 * qk2],
             [2 * qk1 * qk2 - 2 * qk0 * qk3, qk0 ** 2 - qk1 ** 2 + qk2 ** 2 - qk3 ** 2, 2 * qk2 * qk3 + 2 * qk0 * qk1],
             [2 * qk1 * qk3 + 2 * qk0 * qk2, 2 * qk2 * qk3 - 2 * qk0 * qk1, qk0 ** 2 - qk1 ** 2 - qk2 ** 2 + qk3 ** 2]])
        R_glob_sens = R_sens_glob.T
        segment_in_sens = sym.Matrix([sx, sy, sz])
        segment_in_glob = R_glob_sens * segment_in_sens
        H_k = segment_in_glob.jacobian([qk0, qk1, qk2, qk3])
        print('For IMU H_k_vid and h_vid')
        print(segment_in_glob.T)
        print(H_k)

    def init_kalman_static_param(self):
        subject, segment = self.subject, self.segment
        trial_static_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', 'static_back.csv'), index_col=0)
        knee_angles_vicon_static = trial_static_data[TARGETS_LIST[:3]].values

        vid_data_static = pd.read_csv(os.path.join(DATA_PATH, subject, 'triangulated', 'static_back.csv'), index_col=0)
        upper_joint_col = [self.upper_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
        lower_joint_col = [self.lower_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
        segment_in_glob_static = np.mean(vid_data_static[upper_joint_col].values - vid_data_static[lower_joint_col].values, axis=0)
        params_static = SimpleNamespace()
        params_static.R_glob_sens_static = calibrate_segment_in_sensor_frame(
            trial_static_data[['AccelX_R_'+segment, 'AccelY_R_'+segment, 'AccelZ_R_'+segment]].values, segment)
        params_static.segment_length = norm(segment_in_glob_static)
        R_sens_glob_static = params_static.R_glob_sens_static.T
        params_static.segment_in_sens = calibrate_segment_in_global_frame(segment_in_glob_static, R_sens_glob_static)
        print('{} vector in sensor frame: {}'.format(segment, params_static.segment_in_sens))
        return params_static, knee_angles_vicon_static

    def _init_trial_param(self, init_params):
        subject, trial, segment = self.subject, self.trial, self.segment
        upper_joint, lower_joint = self.upper_joint, self.lower_joint
        trial_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', trial+'.csv'), index_col=False)
        knee_angles_vicon = trial_data[TARGETS_LIST[:3]].values
        data_len = trial_data.shape[0]

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

    def update(self, k):
        params = self.params
        gyr, q_esti, acc, P, Q, R_acc_base, R_acc_diff_coeff, V_acc, R_vid_base, V_vid, segment_confidence, segment_in_glob = \
            params.gyr, params.q_esti, params.acc, params.P, params.Q, params.R_acc_base, params.R_acc_diff_coeff,\
            params.V_acc, params.R_vid_base, params.V_vid, params.segment_confidence, params.segment_in_glob
        segment_length = self.params_static.segment_length
        sx, sy, sz = self.params_static.segment_in_sens

        """ a prior system estimation """
        omega = np.array([[0, -gyr[k, 0], -gyr[k, 1], -gyr[k, 2]],
                          [gyr[k, 0], 0, gyr[k, 2], -gyr[k, 1]],
                          [gyr[k, 1], -gyr[k, 2], 0, gyr[k, 0]],
                          [gyr[k, 2], gyr[k, 1], -gyr[k, 0], 0]])
        Ak = np.eye(4) + 0.5 * omega * self.t
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


class KalmanFilterMagIMU:
    # 3. use the madgwick earth magnetic field compensation method
    def __init__(self, subject, segment, trial, init_params):
        self.subject = subject
        self.segment = segment
        self.trial = trial
        self.t = init_params.t
        self.params, self.trial_data = self._init_trial_param(init_params)

    @staticmethod
    def print_h_mat():
        qk0, qk1, qk2, qk3, mx, my, mz = sym.symbols('qk0, qk1, qk2, qk3, mx, my, mz', constant=True)
        R_sens_glob = sym.Matrix([[qk0**2 + qk1**2 - qk2**2 - qk3**2, 2*qk1*qk2+2*qk0*qk3, 2*qk1*qk3-2*qk0*qk2],
                               [2*qk1*qk2-2*qk0*qk3, qk0**2 - qk1**2 + qk2**2 - qk3**2, 2*qk2*qk3+2*qk0*qk1],
                               [2*qk1*qk3+2*qk0*qk2, 2*qk2*qk3-2*qk0*qk1, qk0**2 - qk1**2 - qk2**2 + qk3**2]])
        mag_in_glob = sym.Matrix([mx, my, mz])
        mag_in_sens = R_sens_glob * mag_in_glob
        H_k = mag_in_sens.jacobian([qk0, qk1, qk2, qk3])
        print('For IMU H_k_vid and h_vid')
        print(mag_in_sens.T)
        print(H_k)

    def _init_trial_param(self, init_params):
        subject, trial, segment = self.subject, self.trial, self.segment
        trial_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', trial + '.csv'), index_col=0)
        data_len = trial_data.shape[0]

        params = SimpleNamespace()
        params.q_esti = np.zeros([data_len, 4])
        params.q_esti[0, :] = init_params.quat_init
        params.P = np.zeros([4, 4, data_len])
        params.P[:, :, 0] = np.eye(4)
        params.Q = init_params.gyro_noise * np.eye(4)
        params.acc = trial_data[['AccelX_R_'+segment, 'AccelY_R_'+segment, 'AccelZ_R_'+segment]].values
        params.gyr = np.deg2rad(trial_data[['GyroX_R_'+segment, 'GyroY_R_'+segment, 'GyroZ_R_'+segment]].values)
        params.mag = trial_data[['MagX_R_'+segment, 'MagY_R_'+segment, 'MagZ_R_'+segment]].values

        params.R_acc_diff_coeff = init_params.R_acc_diff_coeff
        params.R_acc_base = np.eye(3) * init_params.acc_noise
        params.R_mag_diff_coeff = init_params.R_mag_diff_coeff
        params.R_mag_base = np.eye(3) * init_params.mag_noise
        params.V_acc = np.eye(3)               # correlation between the acc noise and angular position
        params.V_mag = np.eye(3)
        return params, trial_data

    def update(self, k):
        params = self.params
        gyr, q_esti, acc, P, Q, R_acc_base, R_acc_diff_coeff, V_acc, R_mag_base, R_mag_diff_coeff, V_mag, mag = \
            params.gyr, params.q_esti, params.acc, params.P, params.Q, params.R_acc_base, params.R_acc_diff_coeff,\
            params.V_acc, params.R_mag_base, params.R_mag_diff_coeff, params.V_mag, params.mag

        """ a prior system estimation """
        omega = np.array([[0, -gyr[k, 0], -gyr[k, 1], -gyr[k, 2]],
                          [gyr[k, 0], 0, gyr[k, 2], -gyr[k, 1]],
                          [gyr[k, 1], -gyr[k, 2], 0, gyr[k, 0]],
                          [gyr[k, 2], gyr[k, 1], -gyr[k, 0], 0]])
        Ak = np.eye(4) + 0.5 * omega * self.t
        qk_ = Ak @ q_esti[k-1]
        qk0, qk1, qk2, qk3 = qk_

        P_k_minus = Ak @ P[:, :, k-1] @ Ak.T + Q

        """ correction stage 1, based on acc. Note that the h_mag, z_mag, and R are normalized by the gravity """
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
        z_acc = acc[k, :].T / norm(acc[k, :])
        q_acc_eps = K_k_acc @ (z_acc - h_acc)

        """ correction stage 2, based on mag. Note that the h_mag, z_mag, and R are normalized by the segment length """
        mx, my, mz = TREADMILL_MAG_FIELD
        H_k_mag = np.array([[2*mx*qk0 + 2*my*qk3 - 2*mz*qk2, 2*mx*qk1 + 2*my*qk2 + 2*mz*qk3, -2*mx*qk2 + 2*my*qk1 - 2*mz*qk0, -2*mx*qk3 + 2*my*qk0 + 2*mz*qk1],
                            [-2*mx*qk3 + 2*my*qk0 + 2*mz*qk1, 2*mx*qk2 - 2*my*qk1 + 2*mz*qk0, 2*mx*qk1 + 2*my*qk2 + 2*mz*qk3, -2*mx*qk0 - 2*my*qk3 + 2*mz*qk2],
                            [2*mx*qk2 - 2*my*qk1 + 2*mz*qk0, 2*mx*qk3 - 2*my*qk0 - 2*mz*qk1, 2*mx*qk0 + 2*my*qk3 - 2*mz*qk2, 2*mx*qk1 + 2*my*qk2 + 2*mz*qk3]])
        R_mag = R_mag_base
        K_k_mag = P_k_minus @ H_k_mag.T @ np.matrix(H_k_mag @ P_k_minus @ H_k_mag.T + V_mag @ R_mag @ V_mag.T).I

        h_mag = np.array([mx*(qk0**2 + qk1**2 - qk2**2 - qk3**2) + my*(2*qk0*qk3 + 2*qk1*qk2) + mz*(-2*qk0*qk2 + 2*qk1*qk3),
                          mx*(-2*qk0*qk3 + 2*qk1*qk2) + my*(qk0**2 - qk1**2 + qk2**2 - qk3**2) + mz*(2*qk0*qk1 + 2*qk2*qk3),
                          mx*(2*qk0*qk2 + 2*qk1*qk3) + my*(-2*qk0*qk1 + 2*qk2*qk3) + mz*(qk0**2 - qk1**2 - qk2**2 + qk3**2)])
        z_mag = mag[k, :].T / norm(mag[k, :])
        q_mag_eps = K_k_mag @ (z_mag - h_mag)

        """ combine """
        qk = qk_ + q_acc_eps + q_mag_eps
        q_esti[k, :] = qk / norm(qk)
        P[:, :, k] = (np.eye(4) - K_k_mag @ H_k_mag) @ (np.eye(4) - K_k_acc @ H_k_acc) @ P_k_minus


class VidOnlyKneeAngle:
    @staticmethod
    def get_orientation_from_vectors(segment_z, segment_ml):
        segment_y = np.cross(segment_z, segment_ml)
        segment_x = np.cross(segment_y, segment_z)
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

        q_glob_body = np.array(list(map(temp_fun, R_glob_body)))
        return q_glob_body

    @staticmethod
    def angle_between_vectors(subject, trial):
        vid_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'triangulated', trial + '.csv'), index_col=0)
        joint_col = [joint + '_3d_' + axis for joint in ['RHip', 'RKnee', 'RAnkle'] for axis in ['x', 'y', 'z']]
        joint_hip, joint_knee, joint_ankle = [vid_data[joint_col[3 * i:3 * (i + 1)]].values for i in range(3)]
        shank_y, shank_ml = joint_knee - joint_ankle, [1, 0, 0]
        q_shank_glob_body = VidOnlyKneeAngle.get_orientation_from_vectors(shank_y, shank_ml)
        thigh_y, thigh_ml = joint_hip - joint_knee, [1, 0, 0]
        q_thigh_glob_body = VidOnlyKneeAngle.get_orientation_from_vectors(thigh_y, thigh_ml)
        knee_angles_esti = q_to_knee_angle(q_shank_glob_body, q_thigh_glob_body, np.eye(3), np.eye(3))
        return knee_angles_esti, q_shank_glob_body, q_thigh_glob_body


class MadgwickMagIMU:
    def __init__(self, subject, segment, trial, init_params):
        self.subject = subject
        self.segment = segment
        self.trial = trial
        self.t = init_params.t
        self.beta = 0.1
        self.params, self.trial_data = self._init_trial_param(init_params)

    def _init_trial_param(self, init_params):
        subject, trial, segment = self.subject, self.trial, self.segment
        trial_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', trial + '.csv'), index_col=0)
        data_len = trial_data.shape[0]
        params = SimpleNamespace()
        params.q_esti = np.zeros([data_len, 4])
        params.q_esti[0, :] = init_params.quat_init
        params.acc = trial_data[['AccelX_R_'+segment, 'AccelY_R_'+segment, 'AccelZ_R_'+segment]].values
        params.gyr = np.deg2rad(trial_data[['GyroX_R_'+segment, 'GyroY_R_'+segment, 'GyroZ_R_'+segment]].values)
        params.mag = trial_data[['MagX_R_'+segment, 'MagY_R_'+segment, 'MagZ_R_'+segment]].values
        return params, trial_data

    def update(self, k):
        q, acc, gyr, mag = self.params.q_esti[k-1], self.params.acc[k], self.params.gyr[k], self.params.mag[k]
        acc /= norm(acc)
        mag /= norm(mag)

        # h = qmult(q, qmult((0, mag[0], mag[1], mag[2]), qconjugate(q)))
        # b = np.array([0, norm(h[1:3]), 0, h[3]])
        b = np.concatenate([[0], TREADMILL_MAG_FIELD]) # !!!

        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - acc[0],
            2*(q[0]*q[1] + q[2]*q[3]) - acc[1],
            2*(0.5 - q[1]**2 - q[2]**2) - acc[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - mag[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - mag[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - mag[2]
        ])
        j = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
        ])
        step = j.T.dot(f)
        step /= norm(step)
        qdot = qmult(q, (0, gyr[0], gyr[1], gyr[2])) * 0.5 - self.beta * step.T
        q += qdot * self.t
        self.params.q_esti[k] = q / norm(q)


class MadgwickVidIMU:
    def __init__(self, subject, segment, trial, init_params):
        self.subject = subject
        self.segment = segment
        self.trial = trial
        self.t = init_params.t
        self.beta_acc_coeff = 1     # !!! seems unnecessary?!
        self.beta_vid_coeff = 0.1
        if segment == 'SHANK':
            self.upper_joint, self.lower_joint = 'RKnee', 'RAnkle'
        elif segment == 'THIGH':
            self.upper_joint, self.lower_joint = 'RHip', 'RKnee'
        else:
            raise ValueError('Incorrect segment name')
        self.params, self.trial_data, self.knee_angles_vicon, self.knee_angles_vicon_static = self._init_trial_param(init_params)
        self.R_body_sens = np.eye(3) @ self.params.R_glob_sens_static

    def _init_trial_param(self, init_params):
        subject, trial, segment = self.subject, self.trial, self.segment
        trial_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', trial + '.csv'), index_col=0)
        data_len = trial_data.shape[0]
        knee_angles_vicon = trial_data[TARGETS_LIST[:3]].values
        params = SimpleNamespace()
        params.q_esti = np.zeros([data_len, 4])
        params.q_esti[0, :] = init_params.quat_init
        params.acc = trial_data[['AccelX_R_'+segment, 'AccelY_R_'+segment, 'AccelZ_R_'+segment]].values
        params.gyr = np.deg2rad(trial_data[['GyroX_R_'+segment, 'GyroY_R_'+segment, 'GyroZ_R_'+segment]].values)

        trial_static_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', 'static_back.csv'), index_col=0)
        vid_data_static = pd.read_csv(os.path.join(DATA_PATH, subject, 'triangulated', 'static_back.csv'), index_col=0)
        upper_joint_col = [self.upper_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
        lower_joint_col = [self.lower_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
        segment_in_glob_static = np.mean(
            vid_data_static[upper_joint_col].values - vid_data_static[lower_joint_col].values, axis=0)
        segment_in_glob_static = segment_in_glob_static / norm(segment_in_glob_static)
        params.R_glob_sens_static = calibrate_segment_in_sensor_frame(
            trial_static_data[['AccelX_R_'+segment, 'AccelY_R_'+segment, 'AccelZ_R_'+segment]].values, segment)
        R_sens_glob_static = params.R_glob_sens_static.T
        params.segment_in_sens = calibrate_segment_in_global_frame(segment_in_glob_static, R_sens_glob_static)
        knee_angles_vicon_static = trial_static_data[TARGETS_LIST[:3]].values

        vid_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'triangulated', trial+'.csv'), index_col=0)
        upper_joint_col = [self.upper_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
        lower_joint_col = [self.lower_joint + '_3d_' + axis for axis in ['x', 'y', 'z']]
        segment_in_glob = vid_data[upper_joint_col].values - vid_data[lower_joint_col].values
        params.segment_in_glob = segment_in_glob / norm(segment_in_glob, axis=1).reshape([-1, 1])
        return params, trial_data, knee_angles_vicon, knee_angles_vicon_static

    def update(self, k):
        q, acc, gyr = self.params.q_esti[k-1], self.params.acc[k], self.params.gyr[k]
        sx, sy, sz = self.params.segment_in_sens
        s_glob_x, s_glob_y, s_glob_z = self.params.segment_in_glob[k]
        acc_normed = acc / norm(acc)

        f_acc = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - acc_normed[0],
            2*(q[0]*q[1] + q[2]*q[3]) - acc_normed[1],
            2*(0.5 - q[1]**2 - q[2]**2) - acc_normed[2]])
        J_acc = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0]])
        gradient_dir_acc = np.dot(J_acc.T, f_acc)
        gradient_dir_acc /= norm(gradient_dir_acc).T
        if abs(norm(acc) - GRAVITY) > 2:
            beta_acc = 0
        else:
            beta_acc = self.beta_acc_coeff
        # beta_acc = self.beta_acc_coeff / (1 + norm(rotate_vector(acc, q) - np.array([0, 0, 1]))**2)

        f_vid = np.array([
            sx*(q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2) + sy*(-2*q[0]*q[3] + 2*q[1]*q[2]) + sz*(2*q[0]*q[2] + 2*q[1]*q[3]) - s_glob_x,
            sx*(2*q[0]*q[3] + 2*q[1]*q[2]) + sy*(q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2) + sz*(-2*q[0]*q[1] + 2*q[2]*q[3]) - s_glob_y,
            sx*(-2*q[0]*q[2] + 2*q[1]*q[3]) + sy*(2*q[0]*q[1] + 2*q[2]*q[3]) + sz*(q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2) - s_glob_z])
        J_vid = np.array([
            [2*q[0]*sx + 2*q[2]*sz - 2*q[3]*sy, 2*q[1]*sx + 2*q[2]*sy + 2*q[3]*sz, 2*q[0]*sz + 2*q[1]*sy - 2*q[2]*sx, -2*q[0]*sy + 2*q[1]*sz - 2*q[3]*sx],
            [2*q[0]*sy - 2*q[1]*sz + 2*q[3]*sx, -2*q[0]*sz - 2*q[1]*sy + 2*q[2]*sx, 2*q[1]*sx + 2*q[2]*sy + 2*q[3]*sz, 2*q[0]*sx + 2*q[2]*sz - 2*q[3]*sy],
            [2*q[0]*sz + 2*q[1]*sy - 2*q[2]*sx, 2*q[0]*sy - 2*q[1]*sz + 2*q[3]*sx, -2*q[0]*sx - 2*q[2]*sz + 2*q[3]*sy, 2*q[1]*sx + 2*q[2]*sy + 2*q[3]*sz]])

        gradient_dir_vid = np.dot(J_vid.T, f_vid)
        gradient_dir_vid /= norm(gradient_dir_vid).T
        beta_vid = self.beta_vid_coeff

        qdot = qmult(q, (0, gyr[0], gyr[1], gyr[2])) * 0.5 - beta_acc * gradient_dir_acc - beta_vid * gradient_dir_vid
        q += qdot * self.t
        self.params.q_esti[k] = q / norm(q)





















