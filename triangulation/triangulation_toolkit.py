import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
from const import CAMERA_CALI_DATA_PATH, DATA_PATH, GRAVITY, TARGETS_LIST
from sklearn.metrics import mean_squared_error as mse


def get_camera_mat(images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
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


def compare_axes_results(vicon_data, esti_data, axes=['X', 'Y', 'Z'], start=0, end=1000, title='', ylabel=''):
    print('{:10}'.format(title), end='')
    [print('{:5.1f}\t'.format(np.sqrt(mse(vicon_data[:, i_axis], esti_data[:, i_axis]))), end='')
     for i_axis, axis in enumerate(axes)]
    print()
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('sample')
    for i_axis, axis in enumerate(axes):      # 'Medio-lateral', 'Anterior-posterior', 'Vertical'
        line, = plt.plot(vicon_data[start:end, i_axis], label=axis + ' - Mocap')
        plt.plot(esti_data[start:end, i_axis], '--', color=line.get_color(), label=axis + ' - Smartphone')
        plt.legend()
        plt.grid()


def triangulate(joints, trial_data, camera_pairs_90, camera_pairs_180):
    params_90 = {'init_rot': np.array([[0., 1, 0], [0, 0, -1], [-1, 0, 0]]),
                 'init_tvec': np.array([-900., 1200, 3600]), 'pairs': camera_pairs_90}
    params_180 = {'init_rot': np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]),
                  'init_tvec': np.array([-600., 1200, 2100]), 'pairs': camera_pairs_180}
    data_to_triangluate, projMat = {}, {}
    for camera_, params in zip(['90', '180'], [params_90, params_180]):
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










