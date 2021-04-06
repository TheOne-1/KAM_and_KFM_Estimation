import os

GRAVITY = 9.81
VIDEO_PATH = os.environ.get('VIDEO_DATA_PATH')
OPENPOSE_MODEL_PATH = os.environ.get('OPENPOSE_MODEL_PATH')
VIDEO_ORIGINAL_SAMPLE_RATE = 119.99014859206962
DATA_PATH = os.environ.get('KAM_DATA_PATH')
TRIALS = ['baseline', 'fpa', 'step_width', 'trunk_sway']
TRIALS_PRINT = ['Baseline', 'FPA', 'Step Width', 'Trunk Sway']
STATIC_TRIALS = ['static_back', 'static_side']
SUBJECTS = ['s002_wangdianxin', 's004_ouyangjue', 's005_tangansheng', 's006_xusen', 's007_zuogangao', 's008_liyu',
            's009_sunyubo', 's010_handai', 's011_wuxingze', 's012_likaixiang', 's013_zhangxiaohan', 's014_maqichao',
            's015_weihuan', 's017_tantian', 's018_wangmian', 's019_chenhongyuan', 's020_houjikang'
            # , 's003_linyuan', 's001_tantian', 's016_houjikang'
            ]
# HYPER_PARAM_SUBJECT = ['s003_linyuan']
STEP_TYPES = STANCE, STANCE_SWING = range(2)
STEP_TYPE = STANCE
SEGMENT_DEFINITIONS = {
    'L_FOOT': ['LFCC', 'LFM5', 'LFM2'],
    'R_FOOT': ['RFCC', 'RFM5', 'RFM2'],
    'L_SHANK': ['LTAM', 'LFAL', 'LSK', 'LTT'],
    'R_SHANK': ['RTAM', 'RFAL', 'RSK', 'RTT'],
    'L_THIGH': ['LFME', 'LFLE', 'LTH', 'LFT'],
    'R_THIGH': ['RFME', 'RFLE', 'RTH', 'RFT'],
    'WAIST': ['LIPS', 'RIPS', 'LIAS', 'RIAS'],
    'CHEST': ['MAI', 'SXS', 'SJN', 'CV7', 'LAC', 'RAC']
}
SEGMENT_DATA_FIELDS = [seg_name + '_' + axis for axis in ['X', 'Y', 'Z'] for seg_name in SEGMENT_DEFINITIONS.keys()]
SEGMENT_MASS_PERCENT = {'L_FOOT': 1.37, 'R_FOOT': 1.37, 'R_SHANK': 4.33, 'R_THIGH': 14.16,
                        'WAIST': 11.17, 'CHEST': 15.96, 'L_SHANK': 4.33, 'L_THIGH': 14.16}      # 15.96 + 16.33
SENSOR_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Quat1', 'Quat2',
              'Quat3', 'Quat4']

extract_imu_fields = lambda imus, fields: [field + "_" + imu for imu in imus for field in fields]
extract_video_fields = lambda videos, angles: [video + "_" + position + "_" + angle for video in videos
                                               for position in ["x", "y"] for angle in angles]
VIDEO_LIST = ["LShoulder", "RShoulder", "MidHip", "RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel",
              "LHeel"]
VIDEO_ANGLES = ["90", "180"]

VIDEO_DATA_FIELDS = extract_video_fields(VIDEO_LIST, VIDEO_ANGLES)
IMU_DATA_FIELDS = extract_imu_fields(SENSOR_LIST, IMU_FIELDS)

SAMPLES_BEFORE_STEP = 20
SAMPLES_AFTER_STEP = 20

L_PLATE_FORCE_Z, R_PLATE_FORCE_Z = ['plate_1_force_z', 'plate_2_force_z']

TARGETS_LIST = R_KAM_COLUMN, _, _, _ = ["RIGHT_KNEE_ADDUCTION_MOMENT", "RIGHT_KNEE_FLEXION_MOMENT",
                                        "RIGHT_KNEE_ADDUCTION_ANGLE", "RIGHT_KNEE_ADDUCTION_VELOCITY"]
EXT_KNEE_MOMENT = ['EXT_KM_X', 'EXT_KM_Y', 'EXT_KM_Z']

JOINT_LIST = [marker + '_' + axis for axis in ['X', 'Y', 'Z'] for marker in sum(SEGMENT_DEFINITIONS.values(), [])]

FORCE_DATA_FIELDS = ['plate_' + num + '_' + data_type + '_' + axis for num in ['1', '2']
                     for data_type in ['force', 'cop'] for axis in ['x', 'y', 'z']]

STATIC_DATA = SUBJECT_WEIGHT, SUBJECT_HEIGHT = ['body weight', 'body height']

PHASE_LIST = [EVENT_COLUMN, KAM_PHASE, FORCE_PHASE, STEP_PHASE, SUBJECT_ID, TRIAL_ID] = [
    'Event', 'kam_phase', 'force_phase', 'step_phase', 'subject_id', 'trial_id']
# all the fields of combined data
CONTINUOUS_FIELDS = TARGETS_LIST + EXT_KNEE_MOMENT + IMU_DATA_FIELDS + VIDEO_DATA_FIELDS + FORCE_DATA_FIELDS +\
                    JOINT_LIST + SEGMENT_DATA_FIELDS
DISCRETE_FIELDS = STATIC_DATA + PHASE_LIST
ALL_FIELDS = DISCRETE_FIELDS + CONTINUOUS_FIELDS

RKNEE_MARKER_FIELDS = [marker + axis for marker in ['RFME', 'RFLE'] for axis in ['_X', '_Y', '_Z']]
LEVER_ARM_FIELDS = ['r_x', 'r_y', 'r_z']

FONT_SIZE_LARGE = 24
FONT_SIZE = 20
FONT_SIZE_SMALL = 18
FONT_DICT = {'fontsize': FONT_SIZE, 'fontname': 'Arial'}
FONT_DICT_LARGE = {'fontsize': FONT_SIZE_LARGE, 'fontname': 'Arial'}
FONT_DICT_SMALL = {'fontsize': FONT_SIZE_SMALL, 'fontname': 'Arial'}
FONT_DICT_X_SMALL = {'fontsize': 15, 'fontname': 'Arial'}
LINE_WIDTH = 2
LINE_WIDTH_THICK = 3

SENSOR_COMBINATION = ['8IMU_2camera', '8IMU', '3IMU_2camera', '3IMU', '1IMU_2camera', '1IMU', '2camera']
SENSOR_COMBINATION_SORTED = ['8IMU_2camera', '3IMU_2camera', '8IMU', '1IMU_2camera', '3IMU', '2camera', '1IMU']

EXAMPLE_DATA_FIELDS = [
    'body weight', 'body height', 'force_phase',

    'EXT_KM_X', 'EXT_KM_Y', 'EXT_KM_Z',

    'AccelX_L_FOOT', 'AccelY_L_FOOT', 'AccelZ_L_FOOT', 'GyroX_L_FOOT', 'GyroY_L_FOOT',
    'GyroZ_L_FOOT', 'MagX_L_FOOT', 'MagY_L_FOOT', 'MagZ_L_FOOT', 'Quat1_L_FOOT', 'Quat2_L_FOOT', 'Quat3_L_FOOT',
    'Quat4_L_FOOT', 'AccelX_R_FOOT', 'AccelY_R_FOOT', 'AccelZ_R_FOOT', 'GyroX_R_FOOT', 'GyroY_R_FOOT', 'GyroZ_R_FOOT',
    'MagX_R_FOOT', 'MagY_R_FOOT', 'MagZ_R_FOOT', 'Quat1_R_FOOT', 'Quat2_R_FOOT', 'Quat3_R_FOOT', 'Quat4_R_FOOT',
    'AccelX_R_SHANK', 'AccelY_R_SHANK', 'AccelZ_R_SHANK', 'GyroX_R_SHANK', 'GyroY_R_SHANK', 'GyroZ_R_SHANK',
    'MagX_R_SHANK', 'MagY_R_SHANK', 'MagZ_R_SHANK', 'Quat1_R_SHANK', 'Quat2_R_SHANK', 'Quat3_R_SHANK', 'Quat4_R_SHANK',
    'AccelX_R_THIGH', 'AccelY_R_THIGH', 'AccelZ_R_THIGH', 'GyroX_R_THIGH', 'GyroY_R_THIGH', 'GyroZ_R_THIGH',
    'MagX_R_THIGH', 'MagY_R_THIGH', 'MagZ_R_THIGH', 'Quat1_R_THIGH', 'Quat2_R_THIGH', 'Quat3_R_THIGH', 'Quat4_R_THIGH',
    'AccelX_WAIST', 'AccelY_WAIST', 'AccelZ_WAIST', 'GyroX_WAIST', 'GyroY_WAIST', 'GyroZ_WAIST', 'MagX_WAIST',
    'MagY_WAIST', 'MagZ_WAIST', 'Quat1_WAIST', 'Quat2_WAIST', 'Quat3_WAIST', 'Quat4_WAIST', 'AccelX_CHEST',
    'AccelY_CHEST', 'AccelZ_CHEST', 'GyroX_CHEST', 'GyroY_CHEST', 'GyroZ_CHEST', 'MagX_CHEST', 'MagY_CHEST',
    'MagZ_CHEST', 'Quat1_CHEST', 'Quat2_CHEST', 'Quat3_CHEST', 'Quat4_CHEST', 'AccelX_L_SHANK', 'AccelY_L_SHANK',
    'AccelZ_L_SHANK', 'GyroX_L_SHANK', 'GyroY_L_SHANK', 'GyroZ_L_SHANK', 'MagX_L_SHANK', 'MagY_L_SHANK', 'MagZ_L_SHANK',
    'Quat1_L_SHANK', 'Quat2_L_SHANK', 'Quat3_L_SHANK', 'Quat4_L_SHANK', 'AccelX_L_THIGH', 'AccelY_L_THIGH',
    'AccelZ_L_THIGH', 'GyroX_L_THIGH', 'GyroY_L_THIGH', 'GyroZ_L_THIGH', 'MagX_L_THIGH', 'MagY_L_THIGH', 'MagZ_L_THIGH',
    'Quat1_L_THIGH', 'Quat2_L_THIGH', 'Quat3_L_THIGH', 'Quat4_L_THIGH',

    'LShoulder_x_90', 'LShoulder_x_180',
    'LShoulder_y_90', 'LShoulder_y_180', 'RShoulder_x_90', 'RShoulder_x_180', 'RShoulder_y_90', 'RShoulder_y_180',
    'MidHip_x_90', 'MidHip_x_180', 'MidHip_y_90', 'MidHip_y_180', 'RHip_x_90', 'RHip_x_180', 'RHip_y_90', 'RHip_y_180',
    'LHip_x_90', 'LHip_x_180', 'LHip_y_90', 'LHip_y_180', 'RKnee_x_90', 'RKnee_x_180', 'RKnee_y_90', 'RKnee_y_180',
    'LKnee_x_90', 'LKnee_x_180', 'LKnee_y_90', 'LKnee_y_180', 'RAnkle_x_90', 'RAnkle_x_180', 'RAnkle_y_90',
    'RAnkle_y_180', 'LAnkle_x_90', 'LAnkle_x_180', 'LAnkle_y_90', 'LAnkle_y_180', 'RHeel_x_90', 'RHeel_x_180',
    'RHeel_y_90', 'RHeel_y_180', 'LHeel_x_90', 'LHeel_x_180', 'LHeel_y_90', 'LHeel_y_180',

    'plate_2_force_x', 'plate_2_force_y', 'plate_2_force_z', 'plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z',

    'LFCC_X', 'LFM5_X',
    'LFM2_X', 'RFCC_X', 'RFM5_X', 'RFM2_X', 'LTAM_X', 'LFAL_X', 'LSK_X', 'LTT_X', 'RTAM_X', 'RFAL_X', 'RSK_X', 'RTT_X',
    'LFME_X', 'LFLE_X', 'LTH_X', 'LFT_X', 'RFME_X', 'RFLE_X', 'RTH_X', 'RFT_X', 'LIPS_X', 'RIPS_X', 'LIAS_X', 'RIAS_X',
    'MAI_X', 'SXS_X', 'SJN_X', 'CV7_X', 'LAC_X', 'RAC_X', 'LFCC_Y', 'LFM5_Y', 'LFM2_Y', 'RFCC_Y', 'RFM5_Y', 'RFM2_Y',
    'LTAM_Y', 'LFAL_Y', 'LSK_Y', 'LTT_Y', 'RTAM_Y', 'RFAL_Y', 'RSK_Y', 'RTT_Y', 'LFME_Y', 'LFLE_Y', 'LTH_Y', 'LFT_Y',
    'RFME_Y', 'RFLE_Y', 'RTH_Y', 'RFT_Y', 'LIPS_Y', 'RIPS_Y', 'LIAS_Y', 'RIAS_Y', 'MAI_Y', 'SXS_Y', 'SJN_Y', 'CV7_Y',
    'LAC_Y', 'RAC_Y', 'LFCC_Z', 'LFM5_Z', 'LFM2_Z', 'RFCC_Z', 'RFM5_Z', 'RFM2_Z', 'LTAM_Z', 'LFAL_Z', 'LSK_Z', 'LTT_Z',
    'RTAM_Z', 'RFAL_Z', 'RSK_Z', 'RTT_Z', 'LFME_Z', 'LFLE_Z', 'LTH_Z', 'LFT_Z', 'RFME_Z', 'RFLE_Z', 'RTH_Z', 'RFT_Z',
    'LIPS_Z', 'RIPS_Z', 'LIAS_Z', 'RIAS_Z', 'MAI_Z', 'SXS_Z', 'SJN_Z', 'CV7_Z', 'LAC_Z', 'RAC_Z']
