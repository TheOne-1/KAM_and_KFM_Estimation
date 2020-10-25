import wearable_toolkit
import pandas as pd
import numpy as np
import os

from config import DATA_PATH

SENSOR_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK',
               'L_THIGH']  # this should consistent with Sage script

SEGMENT_DEFITIONS = {
    'L_FOOT': ['LFCC', 'LFM5', 'LFM2'],
    'R_FOOT': ['RFCC', 'RFM5', 'RFM2'],
    'L_SHANK': ['LTAM', 'LFAL', 'LSK', 'LTT'],
    'R_SHANK': ['RTAM', 'RFAL', 'RSK', 'RTT'],
    'L_THIGH': ['LFME', 'LFLE', 'LTH', 'LFT'],
    'R_THIGH': ['RFME', 'RFLE', 'RTH', 'RFT'],
    'PELVIS': ['LIPS', 'RIPS', 'LIAS', 'RIAS'],
    'TRUNK': ['MAI', 'SXS', 'SJN', 'CV7', 'LAC', 'RAC']
}

subjects = ['s001_tantian', 's002_wangdianxin', 's003_linyuan', 's004_ouyangjue', 's005_tangansheng', 's006_xusen',
            's007_zuogangao',
            's008_liyu', 's009_sunyubo', 's010_handai', 's011_wuxingze', 's012_likaixiang', 's013_zhangxiaohan',
            's014_maqichao',
            's015_weihuan', 's016_houjikang']


def sync_and_crop_data_frame(vicon_data_path, imu_data_path, v3d_data_path, video_90_data_path, video_180_data_path,
                             middle_data_path, vicon_calibrate_data_path):
    # read vicon, imu data
    vicon_data = wearable_toolkit.ViconCsvReader(vicon_data_path, SEGMENT_DEFITIONS, vicon_calibrate_data_path)
    video_90_data = wearable_toolkit.VideoCsvReader(video_90_data_path)
    video_180_data = wearable_toolkit.VideoCsvReader(video_180_data_path)
    imu_data = wearable_toolkit.SageCsvReader(imu_data_path)
    V3d_data = wearable_toolkit.Visual3dCsvReader(v3d_data_path)

    # interpolate low probability data
    video_90_data.fill_low_probability_data()
    video_180_data.fill_low_probability_data()

    # create step events
    V3d_data.create_step_id('swing+stance')
    imu_data.create_step_id('swing+stance', False)

    # Synchronize Vicon and IMU data
    vicon_sync_data = vicon_data.get_angular_velocity_theta('R_SHANK')[0:1000]
    imu_sync_data = imu_data.get_norm('R_SHANK', 'Gyro')[0:1000]
    print("vicon-imu synchronization")
    vicon_imu_sync_delay = wearable_toolkit.sync_via_correlation(vicon_sync_data, imu_sync_data, False)

    # Synchronize Vicon and Video data 90
    vicon_sync_data = np.pi / 2 - vicon_data.get_rshank_angle('X')[0:1500]
    vicon_sync_data[np.isnan(vicon_sync_data)] = 0
    video_90_sync_data = np.pi / 2 - video_90_data.get_rshank_angle()[0:1500]
    print("vicon-video_90 synchronization")
    vicon_video_90_sync_delay = wearable_toolkit.sync_via_correlation(-vicon_sync_data, -video_90_sync_data, False)

    # Synchronize Vicon and Video data 180
    vicon_sync_data = np.pi / 2 - vicon_data.get_rshank_angle('Y')[0:1500]
    vicon_sync_data[np.isnan(vicon_sync_data)] = 0
    video_180_sync_data = np.pi / 2 - video_180_data.get_rshank_angle()[0:1500]
    print("vicon-video_180 synchronization")
    vicon_video_180_sync_delay = wearable_toolkit.sync_via_correlation(-vicon_sync_data, -video_180_sync_data, False)

    # Prepare output data
    minimum_delay = min([-imu_data.get_first_event_index(), vicon_imu_sync_delay, vicon_video_90_sync_delay,
                         vicon_video_180_sync_delay])
    vicon_delay = 0 - minimum_delay
    imu_delay = vicon_imu_sync_delay - minimum_delay
    video_90_delay = vicon_video_90_sync_delay - minimum_delay
    video_180_delay = vicon_video_180_sync_delay - minimum_delay

    # crop redundant data
    imu_data.crop(imu_delay)
    vicon_data.crop(vicon_delay)
    V3d_data.crop(vicon_delay)
    video_90_data.crop(video_90_delay)
    video_180_data.crop(video_180_delay)

    # rename video_columns
    video_90_data.data_frame.columns = [col + '_90' for col in video_90_data.data_frame.columns]
    video_180_data.data_frame.columns = [col + '_180' for col in video_180_data.data_frame.columns]
    min_length = min([x.data_frame.shape[0] for x in [imu_data, video_90_data, video_180_data, V3d_data, vicon_data]])
    middle_data = pd.concat(
        [imu_data.data_frame, video_90_data.data_frame, video_180_data.data_frame, V3d_data.data_frame,
         vicon_data.data_frame], axis=1)
    middle_data = middle_data.loc[:min_length]

    # drop missing IMU data steps
    middle_data_tmp = pd.concat([imu_data.data_frame, V3d_data.data_frame], axis=1)
    dropped_steps = (
        middle_data_tmp[(middle_data_tmp.isnull()).any(axis=1)]['Event'].dropna().drop_duplicates()).tolist()
    print("containing corrupted steps: {}".format(dropped_steps))
    for step in dropped_steps:
        middle_data.loc[middle_data['Event'] == step, 'Event'] = -step
    middle_data.to_csv(middle_data_path)


def get_combined_data():
    trials = ['baseline', 'fpa', 'step_width', 'trunk_sway']
    for subject in subjects:
        for trial in trials:
            print("Subject {}, Trial {}".format(subject, trial))
            vicon_data_path = os.path.join(DATA_PATH, subject, 'vicon', trial + '.csv')
            vicon_calibrate_data_path = os.path.join(DATA_PATH, subject, 'vicon', 'calibrate' + '.csv')
            video_data_path_90 = os.path.join(DATA_PATH, subject, 'video_output', trial + '_90.csv')
            video_data_path_180 = os.path.join(DATA_PATH, subject, 'video_output', trial + '_180.csv')
            imu_data_path = os.path.join(DATA_PATH, subject, 'imu', trial + '.csv')
            v3d_data_path = os.path.join(DATA_PATH, subject, 'v3d', trial + '.csv')
            middle_data_path = os.path.join(DATA_PATH, subject, 'combined', trial + '.csv')
            sync_and_crop_data_frame(vicon_data_path, imu_data_path, v3d_data_path, video_data_path_90,
                                     video_data_path_180, middle_data_path, vicon_calibrate_data_path)


def get_static_combined_data():
    trials = ['static_back', 'static_side']
    for subject in subjects:
        for trial in trials:
            vicon_data_path = os.path.join(DATA_PATH, subject, 'vicon', trial + '.csv')
            calibrate_vicon_data_path = os.path.join(DATA_PATH, subject, 'vicon', 'calibrate' + '.csv')
            video_data_path_90 = os.path.join(DATA_PATH, subject, 'video_output', trial + '_90.csv')
            video_data_path_180 = os.path.join(DATA_PATH, subject, 'video_output', trial + '_180.csv')
            imu_data_path = os.path.join(DATA_PATH, subject, 'imu', trial + '.csv')
            middle_data_path = os.path.join(DATA_PATH, subject, 'combined', trial + '.csv')
            vicon_data = wearable_toolkit.ViconCsvReader(vicon_data_path, SEGMENT_DEFITIONS, calibrate_vicon_data_path)
            video_90_data = wearable_toolkit.VideoCsvReader(video_data_path_90)
            video_180_data = wearable_toolkit.VideoCsvReader(video_data_path_180)

            video_90_data.data_frame.columns = [col + '_90' for col in video_90_data.data_frame.columns]
            video_180_data.data_frame.columns = [col + '_180' for col in video_180_data.data_frame.columns]

            imu_data = wearable_toolkit.SageCsvReader(imu_data_path)
            middle_data = pd.concat(
                [imu_data.data_frame.loc[:450], video_90_data.data_frame.loc[:450], video_180_data.data_frame.loc[:450],
                 vicon_data.data_frame.loc[:450]], axis=1)
            middle_data.to_csv(middle_data_path)


if __name__ == "__main__":
    get_static_combined_data()
    get_combined_data()
