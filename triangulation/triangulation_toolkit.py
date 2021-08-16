import numpy as np
import cv2
import matplotlib.pyplot as plt
import sympy as sym
from numpy import cos, sin, arctan2
import os
import glob
from const import CAMERA_CALI_DATA_PATH, IMSHOW_OFFSET
from sklearn.metrics import mean_squared_error as mse
from transforms3d.quaternions import mat2quat



def get_camera_mat(images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 33.33
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            # cv2.imshow('img', cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5))))
            # cv2.waitKey(500)
        else:
            print('Corners not found in {}'.format(fname))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist


def compute_projection_mat(pairs, mtx, dist, init_rot=None, init_tvec=None):
    p_3d_array = np.array([p_3d for (p_3d, p_2d) in pairs.values()], np.float32)
    p_2d_array = np.array([p_2d for (p_3d, p_2d) in pairs.values()], np.float32)
    init_rvec = cv2.Rodrigues(init_rot)[0]
    ret, rvecs, tvecs = cv2.solvePnP(p_3d_array, p_2d_array, mtx, dist, useExtrinsicGuess=True,
                                     rvec=init_rvec, tvec=init_tvec)     # , flags=cv2.SOLVEPNP_AP3P
    rot = cv2.Rodrigues(rvecs)[0]
    # print(tvecs)
    # print(rot)
    mat_projection = np.matmul(mtx, np.column_stack([rot, tvecs]))
    return mat_projection


def extract_a_vid_frame(vid_dir, i_img):
    cap = cv2.VideoCapture(vid_dir)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if 0 > i_img:
        raise ValueError('Only positive i img.')
    elif i_img >= frame_count:
        raise ValueError('i img larger than the video length.')
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_img)
    retrieve_flag, img = cap.read()
    return img


def compute_rmse(data_vicon, data_video, axes=['X', 'Y', 'Z']):
    for i_axis, axis in enumerate(axes):
        rmse = np.sqrt(mse(data_vicon[:, i_axis], data_video[:, i_axis]))
        print(rmse)


def compare_axes_results(vicon_knee, video_knee, axes=['X', 'Y', 'Z'], start=1000, end=2000):
    plt.figure()
    for i_axis, axis in enumerate(axes):      # 'Medio-lateral', 'Anterior-posterior', 'Vertical'
        line, = plt.plot(vicon_knee[start:end, i_axis], label=axis+' - Mocap')
        plt.plot(video_knee[start:end, i_axis], '--', color=line.get_color(), label=axis+' - Smartphone')
        plt.ylabel('Knee center (mm)'.format(axis))
        plt.legend()
        plt.grid()


def triangulate(joints, trial_data, camera_pairs_90, camera_pairs_180):
    params_90 = {'init_rot': np.array([[0., 1, 0], [0, 0, -1], [-1, 0, 0]]),
                 'init_tvec': np.array([-900., 1200, 3600]), 'pairs': camera_pairs_90}
    params_180 = {'init_rot': np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]),
                  'init_tvec': np.array([-600., 1200, 2100]), 'pairs': camera_pairs_180}
    data_to_triangluate, projMat = {}, {}
    for camera_, params in zip(['90', '180'], [params_90, params_180]):
        # print(camera_)
        images = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_ + '_camera_matrix', '*.png'))
        ret, mtx, dist = get_camera_mat(images)
        projMat[camera_] = compute_projection_mat(params['pairs'], mtx, dist, params['init_rot'], params['init_tvec'])
        h, w = cv2.imread(images[0]).shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        for joint in joints:
            op_undist = trial_data[[joint+'_x_'+camera_, joint+'_y_'+camera_]].values.reshape([-1, 1, 2])
            op_undist = cv2.undistortPoints(op_undist, mtx, dist, P=newcameramtx)
            data_to_triangluate[joint+'_'+camera_] = op_undist

    video_triangulated = {}
    for joint in joints:
        video_triangulated_4d = cv2.triangulatePoints(
            projMat['90'], projMat['180'], data_to_triangluate[joint+'_90'], data_to_triangluate[joint+'_180'])
        video_triangulated[joint] = (video_triangulated_4d[:3, :] / video_triangulated_4d[3, :]).T
    return video_triangulated


def calibrate_leg_in_global_frame(knee, ankle, R_glob_to_sens):
    shank_in_sens = R_glob_to_sens.T @ (knee - ankle).T
    return shank_in_sens / np.linalg.norm(shank_in_sens)


def calibrate_segment_in_sensor_frame(acc_static):
    ax, ay, az = np.mean(acc_static, axis=0)
    r = arctan2(ax, ay)
    p = arctan2(-az, cos(r)*ay + sin(r)*ax)
    print('IMU sensor calibration, roll = {:5.2f} deg, pitch = {:5.2f} deg'.
          format(np.rad2deg(r), np.rad2deg(p)))     # to check, r and p should be small
    Cr, Sr, Cp, Sp = sym.symbols('Cr, Sr, Cp, Sp', constant=True)
    R_sens_to_glob = print_orientation_cali_mat()
    R_sens_to_glob = np.array(R_sens_to_glob.subs({Cr: cos(r), Sr: sin(r), Cp: cos(p), Sp: sin(p)})).astype(np.float64)
    return R_sens_to_glob


def print_orientation_cali_mat():
    Cr, Sr, Cp, Sp, ax, ay, az = sym.symbols('Cr, Sr, Cp, Sp, ax, ay, az', constant=True)
    R_coordinate_switch = sym.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    R_pitch = sym.Matrix([[1, 0, 0], [0, Cp, -Sp], [0, Sp, Cp]])
    R_roll = sym.Matrix([[Cr, -Sr, 0], [Sr, Cr, 0], [0, 0, 1]])
    R_sens_to_glob = R_coordinate_switch * R_pitch * R_roll

    # exp = R_sens_to_glob * sym.Matrix([ax, ay, az])
    # collected = sym.expand(exp[1])
    # collected = sym.collect(collected, [Cp, Sp])
    # print('For IMU calibration,', end=' ')
    # print(collected)        # assist my derivation, only need once
    return R_sens_to_glob


def print_h_mat():
    qk0, qk1, qk2, qk3, sx, sy, sz = sym.symbols('qk0, qk1, qk2, qk3, sx, sy, sz', constant=True)
    R_glob_to_sens = sym.Matrix([[qk0**2 + qk1**2 - qk2**2 - qk3**2, 2*qk1*qk2+2*qk0*qk3, 2*qk1*qk3-2*qk0*qk2],
                           [2*qk1*qk2-2*qk0*qk3, qk0**2 - qk1**2 + qk2**2 - qk3**2, 2*qk2*qk3+2*qk0*qk1],
                           [2*qk1*qk3+2*qk0*qk2, 2*qk2*qk3-2*qk0*qk1, qk0**2 - qk1**2 - qk2**2 + qk3**2]])
    shank_in_glob = sym.Matrix([sx, sy, sz])
    shank_in_sens = R_glob_to_sens * shank_in_glob
    H_k = shank_in_sens.jacobian([qk0, qk1, qk2, qk3])
    print('For IMU H_k_vid and h_vid')
    print(shank_in_sens)
    print(H_k)


    temp1 = [[sx*(qk0**2 + qk1**2 - qk2**2 - qk3**2) + sy*(2*qk0*qk3 + 2*qk1*qk2) + sz*(-2*qk0*qk2 + 2*qk1*qk3)],
             [sx*(-2*qk0*qk3 + 2*qk1*qk2) + sy*(qk0**2 - qk1**2 + qk2**2 - qk3**2) + sz*(2*qk0*qk1 + 2*qk2*qk3)],
             [sx*(2*qk0*qk2 + 2*qk1*qk3) + sy*(-2*qk0*qk1 + 2*qk2*qk3) + sz*(qk0**2 - qk1**2 - qk2**2 + qk3**2)]]

    temp2 = [[2*qk0*sx - 2*qk2*sz + 2*qk3*sy, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz, -2*qk0*sz + 2*qk1*sy - 2*qk2*sx, 2*qk0*sy + 2*qk1*sz - 2*qk3*sx],
             [2*qk0*sy + 2*qk1*sz - 2*qk3*sx, 2*qk0*sz - 2*qk1*sy + 2*qk2*sx, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz, -2*qk0*sx + 2*qk2*sz - 2*qk3*sy],
             [2*qk0*sz - 2*qk1*sy + 2*qk2*sx, -2*qk0*sy - 2*qk1*sz + 2*qk3*sx, 2*qk0*sx - 2*qk2*sz + 2*qk3*sy, 2*qk1*sx + 2*qk2*sy + 2*qk3*sz]]


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

        dcm_mat = np.array([segment_x, segment_y, segment_z])
        dcm_mat = np.swapaxes(dcm_mat, 0, 1)

        def temp_fun(dcm_mat):
            if np.isnan(dcm_mat).any():
                return np.array([1, 0, 0, 0])
            else:
                quat = mat2quat(dcm_mat)
                return quat / np.linalg.norm(quat)

        quat_vicon = np.array(list(map(temp_fun, dcm_mat)))

        # for i_axis in range(4):
        #     plt.figure()
        #     plt.plot(quat_vicon[:, i_axis])
        #     plt.plot(data_df['Quat' + str(i_axis+1) + '_R_SHANK'])
        # plt.show()

        return quat_vicon, segment_x, segment_y, segment_z




"""symble_examples"""
def this_is_not_a_function_just_a_example():
    """Below are original manual solutions, not useful for now"""
    """step 3, simplify projection matrix (only translation)"""
    if step == 30:
        u, v, X, Y, Z, fx, fy, u0, v0 = sym.symbols('u, v, X, Y, Z, fx, fy, u0, v0', constant=True)
        k, p, q = sym.symbols('k, p, q', constant=False)
        mat_camera = sym.Matrix([[fx, 0, u0, 0],
                                 [0, fy, v0, 0],
                                 [0, 0, 1, 0]])
        mat_transform = sym.Matrix([[0, 1, 0, k],
                                    [0, 0, -1, p],
                                    [-1, 0, 0, q],
                                    [0, 0, 0, 1]])
        vec_world = sym.Matrix([[X], [Y], [Z], [1]])
        vec_pix = sym.Matrix([[u], [v], [1]])
        exp = mat_camera * mat_transform * vec_world + (X - q) * vec_pix
        for row in exp:
            collected = sym.expand(row)
            collected = sym.collect(collected, [k, p, q])
            print(collected)

    """step 4, solve projection matrix (only translation)"""
    if step == 40:
        number_of_pairs = len(right_camera_pairs[subject])
        images = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_ + '_camera_matrix', '*.png'))
        cv2.waitKey()
        ret, mtx, dist = get_camera_mat(images)

        a, b = np.zeros([2 * number_of_pairs, 3]), np.zeros([2 * number_of_pairs])
        for i_point, (p_3d, p_2d) in enumerate(right_camera_pairs[subject]):
            X, Y, Z = p_3d
            u, v = p_2d
            fx, fy, u0, v0 = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]
            a[2*i_point, :] = [fx, 0, -u + u0]
            a[2*i_point+1, :] = [0, fy, -v + v0]
            b[2*i_point] = - (X*u - X*u0 + Y*fx)
            b[2*i_point+1] = - (X*v - X*v0 - Z*fy)
        p = np.linalg.lstsq(a, b, rcond=-1)
        print(p)



