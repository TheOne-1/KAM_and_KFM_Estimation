import pandas as pd
import numpy as np
import glob
import os
from const import SUBJECTS, TRIALS, DATA_PATH, STATIC_TRIALS
from triangulation.triangulation_toolkit import extract_a_vid_frame, compare_axes_results, triangulate, compute_rmse
from wearable_toolkit import ViconCsvReader
from const import IMSHOW_OFFSET, CAMERA_CALI_DATA_PATH, camera_pairs_all_sub_90, camera_pairs_all_sub_180
import cv2
import matplotlib.pyplot as plt


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

step = 5

"""step 1, extract images from slow-motion video, only do once"""
if step == 1:
    for camera_ in ['90', '180']:
        vid_dir = glob.glob(os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', '*.MOV'))
        for i_vid, vid in enumerate(vid_dir):
            save_dir = os.path.join(CAMERA_CALI_DATA_PATH, camera_+'_camera_matrix', str(i_vid) +'.png')
            img = extract_a_vid_frame(vid, 30)
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
        data, _ = ViconCsvReader.reading(CAMERA_CALI_DATA_PATH + '\\' + file)
        for key in data.keys():
            point_3d_vicon[key] = np.mean(data[key].values, axis=0) - marker_radius * np.array(marker_orientation)
            print('[({:6.1f}, {:6.1f}, {:6.1f}), (., .)],'.format(*point_3d_vicon[key]))

"""step 4, use SOLVEPNP to get camera pose, then triangulate to get knee center"""
if step == 4:
    trial_data_dir = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\KAM\\' + subject + '\combined\\baseline.csv'
    trial_data = pd.read_csv(trial_data_dir, index_col=False)
    video_triangulated = triangulate(['RKnee'], trial_data, camera_pairs_all_sub_90[subject], camera_pairs_all_sub_180[subject])
    video_knee = video_triangulated['RKnee']
    vicon_knee = (trial_data[['RFME_X', 'RFME_Y', 'RFME_Z']].values + trial_data[['RFLE_X', 'RFLE_Y', 'RFLE_Z']].values) / 2
    compare_axes_results(vicon_knee, video_knee)
    compute_rmse(vicon_knee, video_knee)

    # sub_height, sub_weight = trial_data['body height'].iloc[0], trial_data['body weight'].iloc[0]
    # grf = - trial_data[['plate_2_force_x', 'plate_2_force_y', 'plate_2_force_z']].values
    # cop = trial_data[['plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z']].values
    # video_knee_moments = np.cross(cop - video_knee, grf) / (sub_height * sub_weight * 1000.)
    # vicon_knee_moments = trial_data[['EXT_KM_X', 'EXT_KM_Y', 'EXT_KM_Z']].values
    #
    # compare_axes_results(vicon_knee_moments, video_knee_moments, ['X', 'Y'])

"""step 5, save triangulated data """
if step == 5:
    joints = ['RHip', 'RKnee', 'RAnkle']
    for subject in SUBJECTS[:2]:
        os.makedirs(os.path.join(DATA_PATH, subject, 'triangulated'), exist_ok=True)
        for trial in STATIC_TRIALS + TRIALS:
            trial_data_dir = os.path.join(DATA_PATH, subject, 'combined', trial + '.csv')
            trial_data = pd.read_csv(trial_data_dir, index_col=False)
            video_triangulated = triangulate(joints, trial_data, camera_pairs_all_sub_90[subject],
                                             camera_pairs_all_sub_180[subject])
            # for joint in joints:
            video_triangulated_df = pd.DataFrame(np.column_stack([video_triangulated[joint] for joint in joints]))
            video_triangulated_df.columns = [joint + '_3d_' + axis for joint in joints for axis in ['x', 'y', 'z']]
            video_triangulated_df.to_csv(os.path.join(DATA_PATH, subject, 'triangulated', trial+'.csv'))







plt.show()

cv2.destroyAllWindows()

