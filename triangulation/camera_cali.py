import numpy as np
import pandas as pd
import glob
import os
import cv2
from const import SUBJECTS
import matplotlib.pyplot as plt
import sympy as sym
from wearable_toolkit import ViconCsvReader


CAMERA_CALI_DATA_PATH = 'D:\Tian\Research\Projects\VideoIMUCombined\\triangulation\camera cali'
IMSHOW_OFFSET = 1300

right_camera_pairs = {
    's002_wangdianxin': [
        [(  33.9,  174.0,   -0.5), (202., 1479.)],
        [( 560.5,   83.7,  -11.1), (106., 1585.)],
        [(1084.2,  175.3,   -2.5), (79., 1725.)],
        [( 552.4, 1519.8,  -18.6), (865., 1579.)],
        [( 215.7, 1720.9, 2868.1), (919., 157.)],
        [( 214.3, 2028.1, 2862.0), (1064., 160.)],
        [(1110., 778, -117), (487., 1904)]          # 9, tape measured, !!! USE VICON TO MEASURE AGAIN
    ]}

back_camera_pairs = {
    's002_wangdianxin': [
        [(  33.9,  174.0,   -0.5), (153., 1729.)],
        [( 560.5,   83.7,  -11.1), (493., 1763.)],
        [(1084.2,  175.3,   -2.5), (835., 1723.)],
        [( 552.4, 1519.8,  -18.6), (504., 1433.)],
        [( 215.7, 1720.9, 2868.1), (368., 273.)],
        [( 214.3, 2028.1, 2862.0), (378., 321.)],
        [(1425.8, 2030.8, 2856.1), (822., 323.)],
        [(1421.9, 1726.0, 2861.8), (844., 275.)]
    ]}

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


def compute_projection_mat(pairs, mtx, dist):
    p_3d_array = np.array([p_3d for (p_3d, p_2d) in pairs[subject]], np.float32)
    p_2d_array = np.array([p_2d for (p_3d, p_2d) in pairs[subject]], np.float32)
    ret, rvecs, tvecs = cv2.solvePnP(p_3d_array, p_2d_array, mtx, dist)     # , flags=cv2.SOLVEPNP_AP3P
    rot = cv2.Rodrigues(rvecs)[0]
    # print(dist)
    print(tvecs)
    print(rot)
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


def on_click_bottom(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        y = y + IMSHOW_OFFSET
        xy = "%d,%d" % (x, y)
        img_copy = np.copy(img)
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), thickness=-1)
        cv2.putText(img_copy, xy, (50, IMSHOW_OFFSET+50), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), thickness=2)
        cv2.imshow('bottom', img_copy[IMSHOW_OFFSET:, :, :])
        print(x, y)


def on_click_top(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        y = y
        xy = "%d,%d" % (x, y)
        img_copy = np.copy(img)
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), thickness=-1)
        cv2.putText(img_copy, xy, (50, 50), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), thickness=2)
        cv2.imshow('top', img_copy[:500, :, :])
        print(x, y)


subject = SUBJECTS[0]

step = 4
"""step 1, extract images from slow-motion video, only do once"""
if step == 1:
    camera_ = 'back'
    for i_frame in range(200, 1600, 100):
        vid_dir = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', '*.MOV'))[0]
        save_dir = os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', str(i_frame)+'.png')
        img = extract_a_vid_frame(vid_dir, i_frame)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(save_dir, img)

"""step 2, get 2d points from baseline trial pic"""
if step == 2:
    camera_ = 'right'
    if camera_ == 'right': angle = '90.MOV'
    else: angle = '180.MOV'
    vid_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\\video\\' + subject + '\\baseline_' + angle
    img = extract_a_vid_frame(vid_dir, 1000)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('top', img[:500, :, :])
    cv2.setMouseCallback('top', on_click_top)
    cv2.imshow('bottom', img[IMSHOW_OFFSET:, :, :])
    cv2.setMouseCallback('bottom', on_click_bottom)
    cv2.waitKey(0)

"""step 3, get 3d points from vicon"""
if step == 3:
    point_3d_vicon = {}
    marker_radius = 7
    for file, marker_orientation in zip(['points_treadmill.csv', 'points_light.csv'], [[0, 0, 1], [0, 0, -1]]):
        data, _ = ViconCsvReader.reading(file)
        for key in data.keys():
            point_3d_vicon[key] = np.mean(data[key].values, axis=0) - marker_radius * np.array(marker_orientation)
            print('[({:6.1f}, {:6.1f}, {:6.1f}), (., .)],'.format(*point_3d_vicon[key]))

"""step 4, use SOLVEPNP to get camera pose, then triangulate"""
if step == 4:
    trial_data_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\' + subject + '\combined\\baseline.csv'
    trial_data = pd.read_csv(trial_data_dir, index_col=False)
    vid_col = {'right': ['RKnee_x_90', 'RKnee_y_90'], 'back': ['RKnee_x_180', 'RKnee_y_180']}
    data_to_triangluate, projMat = {}, {}
    for camera_, pairs in zip(['right', 'back'], [right_camera_pairs, back_camera_pairs]):
        print(camera_)
        images = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_ + '_camera_matrix', '*.png'))
        ret, mtx, dist = get_camera_mat(images)
        projMat[camera_] = compute_projection_mat(pairs, mtx, dist)
        h, w = cv2.imread(images[0]).shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        op_undist = trial_data[vid_col[camera_]].values.reshape([-1, 1, 2])
        op_undist = cv2.undistortPoints(op_undist, mtx, dist, P=newcameramtx)
        data_to_triangluate[camera_] = op_undist

    video_knee_4d = cv2.triangulatePoints(projMat['right'], projMat['back'], data_to_triangluate['right'], data_to_triangluate['back'])
    video_knee = (video_knee_4d[:3, :] / video_knee_4d[3, :]).T

    vicon_knee = (trial_data[['RFME_X', 'RFME_Y', 'RFME_Z']].values + trial_data[['RFLE_X', 'RFLE_Y', 'RFLE_Z']].values) / 2

    plt.figure()
    for i_axis, axis in enumerate(['X', 'Y', 'Z']):      # 'Medio-lateral', 'Anterior-posterior', 'Vertical'
        line, = plt.plot(vicon_knee[1000:1800, i_axis], label=axis+' - Mocap')
        plt.plot(video_knee[1000:1800, i_axis], '--', color=line.get_color(), label=axis+' - Smartphone')
        plt.ylabel('Knee center (mm)'.format(axis))
        plt.legend()
        plt.grid()
    plt.show()

cv2.destroyAllWindows()

