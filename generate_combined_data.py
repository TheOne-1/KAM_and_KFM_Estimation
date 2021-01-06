import wearable_toolkit
import pandas as pd
import numpy as np
import os
from const import SEGMENT_DEFINITIONS, SUBJECTS, STATIC_TRIALS, TRIALS, DATA_PATH, SUBJECT_HEIGHT, SUBJECT_WEIGHT, \
    SUBJECT_ID, TRIAL_ID

subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)


def sync_and_crop_data_frame(subject, trial):
    vicon_data_path = os.path.join(DATA_PATH, subject, 'vicon', trial + '.csv')
    vicon_calibrate_data_path = os.path.join(DATA_PATH, subject, 'vicon', 'calibrate' + '.csv')
    video_90_data_path = os.path.join(DATA_PATH, subject, 'raw_video_output', trial + '_90.csv')
    video_180_data_path = os.path.join(DATA_PATH, subject, 'raw_video_output', trial + '_180.csv')
    imu_data_path = os.path.join(DATA_PATH, subject, 'imu', trial + '.csv')
    v3d_data_path = os.path.join(DATA_PATH, subject, 'v3d', trial + '.csv')
    middle_data_path = os.path.join(DATA_PATH, subject, 'combined', trial + '.csv')
    is_verbose = False
    # read vicon, imu data
    subject_info = subject_infos.loc[subject, :]
    vicon_data = wearable_toolkit.ViconCsvReader(vicon_data_path, SEGMENT_DEFINITIONS, vicon_calibrate_data_path, subject_info)
    video_90_data = wearable_toolkit.VideoCsvReader(video_90_data_path)
    video_180_data = wearable_toolkit.VideoCsvReader(video_180_data_path)
    imu_data = wearable_toolkit.SageCsvReader(imu_data_path)
    V3d_data = wearable_toolkit.Visual3dCsvReader(v3d_data_path)

    # interpolate low probability data
    video_90_data.fill_low_probability_data()
    video_180_data.fill_low_probability_data()
    video_90_data.low_pass_filtering(15, 100, 2)
    video_180_data.low_pass_filtering(15, 100, 2)
    video_90_data.resample_to_100hz()
    video_180_data.resample_to_100hz()

    # create step events
    imu_data.create_step_id('R_FOOT', verbose=False)

    # Synchronize Vicon and IMU data
    vicon_sync_data = vicon_data.get_angular_velocity_theta('R_SHANK', 1000)
    imu_sync_data = imu_data.get_norm('R_SHANK', 'Gyro')[0:1000]
    print("vicon-imu synchronization")
    vicon_imu_sync_delay = wearable_toolkit.sync_via_correlation(vicon_sync_data, imu_sync_data, is_verbose)

    # Synchronize Vicon and Video data 90
    vicon_sync_data = np.pi / 2 - vicon_data.get_rshank_angle('X')[0:1500]
    vicon_sync_data[np.isnan(vicon_sync_data)] = 0
    video_90_sync_data = np.pi / 2 - video_90_data.get_rshank_angle()[0:1500]
    print("vicon-video_90 synchronization")
    vicon_video_90_sync_delay = wearable_toolkit.sync_via_correlation(-vicon_sync_data, -video_90_sync_data, is_verbose)

    # Synchronize Vicon and Video data 180
    vicon_sync_data = np.pi / 2 - vicon_data.get_rshank_angle('Y')[0:1500]
    vicon_sync_data[np.isnan(vicon_sync_data)] = 0
    video_180_sync_data = np.pi / 2 - video_180_data.get_rshank_angle()[0:1500]
    print("vicon-video_180 synchronization")
    vicon_video_180_sync_delay = wearable_toolkit.sync_via_correlation(-vicon_sync_data, -video_180_sync_data, is_verbose)

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
    middle_data[SUBJECT_HEIGHT] = subject_info[SUBJECT_HEIGHT]
    middle_data[SUBJECT_WEIGHT] = subject_info[SUBJECT_WEIGHT]
    middle_data[SUBJECT_ID] = SUBJECTS.index(subject)
    middle_data[TRIAL_ID] = TRIALS.index(trial)
    middle_data.to_csv(middle_data_path)


def get_combined_data():
    for subject in SUBJECTS:
        for trial in TRIALS:
            print("Subject {}, Trial {}".format(subject, trial))
            sync_and_crop_data_frame(subject, trial)


def get_static_combined_data():
    for subject in SUBJECTS:
        for trial in STATIC_TRIALS:
            vicon_data_path = os.path.join(DATA_PATH, subject, 'vicon', trial + '.csv')
            calibrate_vicon_data_path = os.path.join(DATA_PATH, subject, 'vicon', 'calibrate' + '.csv')
            video_data_path_90 = os.path.join(DATA_PATH, subject, 'raw_video_output', trial + '_90.csv')
            video_data_path_180 = os.path.join(DATA_PATH, subject, 'raw_video_output', trial + '_180.csv')
            imu_data_path = os.path.join(DATA_PATH, subject, 'imu', trial + '.csv')
            middle_data_path = os.path.join(DATA_PATH, subject, 'combined', trial + '.csv')
            subject_info = subject_infos.loc[subject, :]
            vicon_data = wearable_toolkit.ViconCsvReader(vicon_data_path, SEGMENT_DEFINITIONS, calibrate_vicon_data_path, subject_info)
            video_90_data = wearable_toolkit.VideoCsvReader(video_data_path_90)
            video_180_data = wearable_toolkit.VideoCsvReader(video_data_path_180)
            video_90_data.fill_low_probability_data()
            video_180_data.fill_low_probability_data()
            video_90_data.low_pass_filtering(15, 100, 2)
            video_180_data.low_pass_filtering(15, 100, 2)
            video_90_data.resample_to_100hz()
            video_180_data.resample_to_100hz()

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
