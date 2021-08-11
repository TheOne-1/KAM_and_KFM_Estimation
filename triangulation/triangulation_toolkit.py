import numpy as np
import cv2
import matplotlib.pyplot as plt

IMSHOW_OFFSET = 1300

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


def compute_projection_mat(subject, pairs, mtx, dist, init_rot=None, init_tvec=None):
    p_3d_array = np.array([p_3d for (p_3d, p_2d) in pairs[subject].values()], np.float32)
    p_2d_array = np.array([p_2d for (p_3d, p_2d) in pairs[subject].values()], np.float32)
    init_rvec = cv2.Rodrigues(init_rot)[0]
    ret, rvecs, tvecs = cv2.solvePnP(p_3d_array, p_2d_array, mtx, dist, useExtrinsicGuess=True,
                                     rvec=init_rvec, tvec=init_tvec)     # , flags=cv2.SOLVEPNP_AP3P
    rot = cv2.Rodrigues(rvecs)[0]
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


def compare_axes_results(vicon_knee, video_knee, axes=['X', 'Y', 'Z'], start=1000, end=2000):
    plt.figure()
    for i_axis, axis in enumerate(axes):      # 'Medio-lateral', 'Anterior-posterior', 'Vertical'
        line, = plt.plot(vicon_knee[start:end, i_axis], label=axis+' - Mocap')
        plt.plot(video_knee[start:end, i_axis], '--', color=line.get_color(), label=axis+' - Smartphone')
        plt.ylabel('Knee center (mm)'.format(axis))
        plt.legend()
        plt.grid()
