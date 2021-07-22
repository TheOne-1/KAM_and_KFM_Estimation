import numpy as np
import glob
import os
import cv2
from const import SUBJECTS
import matplotlib.pyplot as plt
import sympy as sym


CAMERA_CALI_DATA_PATH = 'D:\Tian\Research\Projects\VideoIMUCombined\\triangulation\camera cali'
IMSHOW_OFFSET = 1300

right_camera_pairs = {
    's002_wangdianxin': [[(-9., -121, -11), (84., 1472)],
                         [(-9., 1675, -11), (882., 1468)],
                         [(1110., 1675, -11), (1066., 1724)],
                         [(1110, 778, -117), (487, 1904)]
                         ]
}


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

        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)      # TODO: shoot another cali video and remove this

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            plt.imshow(img, interpolation='none')

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            # cv2.imshow('img', cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5))))
            # cv2.waitKey(500)
        else:
            print('Corners not found in {}'.format(fname))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist


def compute_projection_mat():
    pass


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


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        y = y + IMSHOW_OFFSET
        xy = "%d,%d" % (x, y)
        img_copy = np.copy(img)
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), thickness=-1)
        cv2.putText(img_copy, xy, (50, IMSHOW_OFFSET+50), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), thickness=2)
        cv2.imshow('', img_copy[IMSHOW_OFFSET:, :, :])
        print(x, y)


camera_ = 'right'
subject = SUBJECTS[0]
if camera_ == 'right': angle = '90.MOV'
else: angle = '180.MOV'

step = 3
"""step 1, extract images from slow-motion video, only do once"""
if step == 1:
    for i_frame in range(200, 1600, 100):
        vid_dir = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', '*.MOV'))[0]
        save_dir = os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', str(i_frame)+'.png')
        img = extract_a_vid_frame(vid_dir, i_frame)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(save_dir, img)

"""step 2, get 2d dots"""
if step == 2:
    vid_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\\video\\' + subject + '\\baseline_' + angle
    img = extract_a_vid_frame(vid_dir, 0)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('', img[IMSHOW_OFFSET:, :, :])
    cv2.setMouseCallback('', on_click)
    cv2.waitKey(0)

"""step 3, simplify projection matrix"""
if step == 3:
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

"""step 4, solve projection matrix"""
if step == 4:
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


"""step 2, get projection matrix"""

cv2.destroyAllWindows()
