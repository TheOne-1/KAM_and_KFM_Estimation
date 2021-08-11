import pandas as pd
import glob
import os
from const import SUBJECTS
from triangulation.triangulation_toolkit import *
from wearable_toolkit import ViconCsvReader



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

CAMERA_CALI_DATA_PATH = 'D:\Tian\Research\Projects\VideoIMUCombined\\triangulation\camera cali'

camera_pairs_90 = {
    's002_wangdianxin': {
        '1': [(  33.9,  174.0,   -0.5), (202., 1479.)],       # 1
        '2': [( 560.5,   83.7,  -11.1), (106., 1585.)],       # 2
        '3': [(1084.2,  175.3,   -2.5), (79., 1725.)],       # 3
        '4': [( 552.4, 1519.8,  -18.6), (865., 1579.)],       # 4
        '5': [( 215.7, 1720.9, 2868.1), (919., 157.)],       # 5
        '6': [( 214.3, 2028.1, 2862.0), (1064., 160.)],       # 6
        '9': [(-1068.0, 1737.1, 2871.4), (797., 373.)],
        '10': [(-1065.8, 2051.7, 2891.2), (899., 368.)],
        },
    's004_ouyangjue': {
        '1': [(  33.9,  174.0,   -0.5), (190., 1485.)],
        '2': [( 560.5,   83.7,  -11.1), (90., 1591.)],
        '3': [(1084.2,  175.3,   -2.5), (66., 1731.)],
        '4': [( 552.4, 1519.8,  -18.6), (852., 1581.)],
        '5': [( 215.7, 1720.9, 2868.1), (910., 162.)],
        '6': [( 214.3, 2028.1, 2862.0), (1052., 168.)],
        '9': [(-1068.0, 1737.1, 2871.4), (797., 374.)],
        '10': [(-1065.8, 2051.7, 2891.2), (897., 371.)],
        },
    }

camera_pairs_180 = {
    's002_wangdianxin': {
        '1': [(  33.9,  174.0,   -0.5), (153., 1729.)],       # 1
        '2': [( 560.5,   83.7,  -11.1), (493., 1763.)],       # 2
        '3': [(1084.2,  175.3,   -2.5), (835., 1723.)],       # 3
        '4': [( 552.4, 1519.8,  -18.6), (504., 1433.)],       # 4
        '5': [( 215.7, 1720.9, 2868.1), (368., 273.)],       # 5
        '6': [( 214.3, 2028.1, 2862.0), (378., 321.)],       # 6
        '7': [(1425.8, 2030.8, 2856.1), (822., 323.)],       # 7
        '8': [(1421.9, 1726.0, 2861.8), (844., 275.)]       # 8
        },
    's004_ouyangjue': {
        '1': [(  33.9,  174.0,   -0.5), (135., 1728.)],
        '2': [( 560.5,   83.7,  -11.1), (476., 1761.)],
        '3': [(1084.2,  175.3,   -2.5), (816., 1719.)],
        '4': [( 552.4, 1519.8,  -18.6), (486., 1433.)],
        '5': [( 215.7, 1720.9, 2868.1), (351., 287.)],
        '6': [( 214.3, 2028.1, 2862.0), (364., 316.)],
        '7': [(1425.8, 2030.8, 2856.1), (806., 323.)],
        '8': [(1421.9, 1726.0, 2861.8), (828., 275.)]
        },
    }

subject = SUBJECTS[0]

step = 4
"""step 1, extract images from slow-motion video, only do once"""
if step == 1:
    camera_ = '180'
    for i_frame in range(200, 1600, 100):
        vid_dir = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', '*.MOV'))[0]
        save_dir = os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', str(i_frame)+'.png')
        img = extract_a_vid_frame(vid_dir, i_frame)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(save_dir, img)

"""step 2, get 2d points from baseline trial pic"""
if step == 2:
    camera_ = '90'
    angle = camera_ + '.MOV'
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
    params_90 = {'vid_col': ['RKnee_x_90', 'RKnee_y_90'], 'init_rot': np.array([[0., 1, 0], [0, 0, -1], [-1, 0, 0]]),
                 'init_tvec': np.array([-900., 1200, 3600]), 'pairs': camera_pairs_90}
    params_180 = {'vid_col': ['RKnee_x_180', 'RKnee_y_180'], 'init_rot': np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]),
                  'init_tvec': np.array([-600., 1200, 2100]), 'pairs': camera_pairs_180}
    data_to_triangluate, projMat = {}, {}
    for camera_, params in zip(['90', '180'], [params_90, params_180]):
        print(camera_)
        images = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_ + '_camera_matrix', '*.png'))
        ret, mtx, dist = get_camera_mat(images)
        projMat[camera_] = compute_projection_mat(subject, params['pairs'], mtx, dist, params['init_rot'], params['init_tvec'])
        h, w = cv2.imread(images[0]).shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        op_undist = trial_data[params['vid_col']].values.reshape([-1, 1, 2])
        op_undist = cv2.undistortPoints(op_undist, mtx, dist, P=newcameramtx)
        data_to_triangluate[camera_] = op_undist

    video_knee_4d = cv2.triangulatePoints(projMat['90'], projMat['180'], data_to_triangluate['90'], data_to_triangluate['180'])
    video_knee = (video_knee_4d[:3, :] / video_knee_4d[3, :]).T

    # video_knee[:, 1] = video_knee[:, 1] + 40            # !!!

    sub_height, sub_weight = trial_data['body height'].iloc[0], trial_data['body weight'].iloc[0]
    grf = - trial_data[['plate_2_force_x', 'plate_2_force_y', 'plate_2_force_z']].values
    cop = trial_data[['plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z']].values
    video_knee_moments = np.cross(cop - video_knee, grf) / (sub_height * sub_weight * 1000.)
    vicon_knee_moments = trial_data[['EXT_KM_X', 'EXT_KM_Y', 'EXT_KM_Z']].values

    compare_axes_results(vicon_knee_moments, video_knee_moments, ['X', 'Y'])


    vicon_knee = (trial_data[['RFME_X', 'RFME_Y', 'RFME_Z']].values + trial_data[['RFLE_X', 'RFLE_Y', 'RFLE_Z']].values) / 2
    compare_axes_results(vicon_knee, video_knee)







plt.show()

cv2.destroyAllWindows()

