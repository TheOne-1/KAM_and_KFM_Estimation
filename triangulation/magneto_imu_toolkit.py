import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from numpy import cos, sin, arctan2, arcsin
import os
import glob
from const import TREADMILL_MAG_FIELD, DATA_PATH, GRAVITY, TARGETS_LIST
from types import SimpleNamespace
from transforms3d.quaternions import rotate_vector, mat2quat, quat2mat
from transforms3d.euler import mat2euler


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


def init_kalman_param(subject, trial, segment, init_params):
    trial_data_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\'\
                     + subject + '\combined\\' + trial + '.csv'
    trial_data = pd.read_csv(trial_data_dir, index_col=False)
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
    params.mag = trial_data[['MagX_R_'+segment, 'MagY_R_'+segment, 'MagZ_R_'+segment]].values

    params.R_acc_diff_coeff = init_params.R_acc_diff_coeff
    params.R_acc_base = np.eye(3) * init_params.acc_noise
    params.R_mag_diff_coeff = init_params.R_mag_diff_coeff
    params.R_mag_base = np.eye(3) * init_params.mag_noise
    params.V_acc = np.eye(3)               # correlation between the acc noise and angular position
    params.V_mag = np.eye(3)
    return params, trial_data, knee_angles_vicon


def update_kalman(params, temp, T, k):
    gyr, q_esti, acc, P, Q, R_acc_base, R_acc_diff_coeff, V_acc, R_mag_base, R_mag_diff_coeff, V_mag, mag = \
        params.gyr, params.q_esti, params.acc, params.P, params.Q, params.R_acc_base, params.R_acc_diff_coeff,\
        params.V_acc, params.R_mag_base, params.R_mag_diff_coeff, params.V_mag, params.mag

    """ a prior system estimation """
    omega = np.array([[0, -gyr[k, 0], -gyr[k, 1], -gyr[k, 2]],
                      [gyr[k, 0], 0, gyr[k, 2], -gyr[k, 1]],
                      [gyr[k, 1], -gyr[k, 2], 0, gyr[k, 0]],
                      [gyr[k, 2], gyr[k, 1], -gyr[k, 0], 0]])
    Ak = np.eye(4) + 0.5 * omega * T
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


