import os
import numpy as np

GRAVITY = 9.81
VIDEO_PATH = os.environ.get('VIDEO_PATH')
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
USED_KEYPOINTS = ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
VIDEO_ANGLES = ["90", "180"]

VIDEO_DATA_FIELDS = extract_video_fields(VIDEO_LIST, VIDEO_ANGLES)
IMU_DATA_FIELDS = extract_imu_fields(SENSOR_LIST, IMU_FIELDS)

SAMPLES_BEFORE_STEP = 20
SAMPLES_AFTER_STEP = 20

L_PLATE_FORCE_Z, R_PLATE_FORCE_Z = ['plate_1_force_z', 'plate_2_force_z']

TARGETS_LIST = _, _, _, _, R_KAM_COLUMN, _, _ = [
    "RIGHT_KNEE_FLEXION_ANGLE", "RIGHT_KNEE_ADDUCTION_ANGLE", "RIGHT_KNEE_INTERNAL_ANGLE",
    "RIGHT_KNEE_FLEXION_MOMENT", "RIGHT_KNEE_ADDUCTION_MOMENT", "RIGHT_KNEE_INTERNAL_MOMENT",
    "RIGHT_KNEE_ADDUCTION_VELOCITY"]
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

CAMERA_CALI_DATA_PATH = 'J:\Projects\VideoIMUCombined\experiment_data\camera cali'
IMSHOW_OFFSET = 1300


camera_pairs_all_sub_90 = {
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
    's005_tangansheng': {
        '1': [(  33.9,  174.0,   -0.5), (196., 1479.)],
        '2': [( 560.5,   83.7,  -11.1), (99., 1585.)],
        '3': [(1084.2,  175.3,   -2.5), (73., 1723.)],
        '4': [( 552.4, 1519.8,  -18.6), (855., 1575.)],
        '5': [( 215.7, 1720.9, 2868.1), (913., 160.)],
        '6': [( 214.3, 2028.1, 2862.0), (1500., 160.)],
        '9': [(-1068.0, 1737.1, 2871.4), (800., 372.)],
        '10': [(-1065.8, 2051.7, 2891.2), (899., 368.)],
        },
    's006_xusen': {
        '1': [(  33.9,  174.0,   -0.5), (195., 1481.)],
        '2': [( 560.5,   83.7,  -11.1), (99., 1587.)],
        '3': [(1084.2,  175.3,   -2.5), (71., 1726.)],
        '4': [( 552.4, 1519.8,  -18.6), (854., 1577.)],
        '5': [( 215.7, 1720.9, 2868.1), (913., 162.)],
        '6': [( 214.3, 2028.1, 2862.0), (1053., 168.)],
        '9': [(-1068.0, 1737.1, 2871.4), (799., 374.)],
        '10': [(-1065.8, 2051.7, 2891.2), (899., 372.)],
        },
    's007_zuogangao': {
        '1': [(  33.9,  174.0,   -0.5), (203., 1478.)],
        '2': [( 560.5,   83.7,  -11.1), (103., 1583.)],
        '3': [(1084.2,  175.3,   -2.5), (77., 1721.)],
        '4': [( 552.4, 1519.8,  -18.6), (859., 1575.)],
        '5': [( 215.7, 1720.9, 2868.1), (918., 159.)],
        '6': [( 214.3, 2028.1, 2862.0), (1056., 163.)],
        '9': [(-1068.0, 1737.1, 2871.4), (806., 372.)],
        '10': [(-1065.8, 2051.7, 2891.2), (905., 367.)],
        },
    's008_liyu': {
        '1': [(  33.9,  174.0,   -0.5), (200., 1481.)],
        '2': [( 560.5,   83.7,  -11.1), (102., 1587.)],
        '3': [(1084.2,  175.3,   -2.5), (75., 1726.)],
        '4': [( 552.4, 1519.8,  -18.6), (855., 1578.)],
        '5': [( 215.7, 1720.9, 2868.1), (917., 162.)],
        '6': [( 214.3, 2028.1, 2862.0), (1055., 169.)],
        '9': [(-1068.0, 1737.1, 2871.4), (804., 376.)],
        '10': [(-1065.8, 2051.7, 2891.2), (902., 373.)],
        },
    's009_sunyubo': {
        '1': [(  33.9,  174.0,   -0.5), (185., 1486.)],
        '2': [( 560.5,   83.7,  -11.1), (88., 1592.)],
        '3': [(1084.2,  175.3,   -2.5), (60., 1732.)],
        '4': [( 552.4, 1519.8,  -18.6), (844., 1582.)],
        '5': [( 215.7, 1720.9, 2868.1), (902., 167.)],
        '6': [( 214.3, 2028.1, 2862.0), (1040., 173.)],
        '9': [(-1068.0, 1737.1, 2871.4), (792., 378.)],
        '10': [(-1065.8, 2051.7, 2891.2), (890., 377.)],
        },
    's010_handai': {
        '1': [(  33.9,  174.0,   -0.5), (198., 1481.)],
        '2': [( 560.5,   83.7,  -11.1), (102., 1587.)],
        '3': [(1084.2,  175.3,   -2.5), (74., 1727.)],
        '4': [( 552.4, 1519.8,  -18.6), (857., 1580.)],
        '5': [( 215.7, 1720.9, 2868.1), (916., 159.)],
        '6': [( 214.3, 2028.1, 2862.0), (1054., 166.)],
        '9': [(-1068.0, 1737.1, 2871.4), (805., 374.)],
        '10': [(-1065.8, 2051.7, 2891.2), (903., 373.)],
        },
    's011_wuxingze': {
        },
    's012_likaixiang': {
        },
    's013_zhangxiaohan': {
        },
    's014_maqichao': {
        },
    's015_weihuan': {
        },
    's017_tantian': {
        },
    's018_wangmian': {
        },
    's019_chenhongyuan': {
        },
    's020_houjikang': {
        }
    }

camera_pairs_all_sub_180 = {
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
    's005_tangansheng': {
        '1': [(  33.9,  174.0,   -0.5), (150., 1730.)],
        '2': [( 560.5,   83.7,  -11.1), (489., 1765.)],
        '3': [(1084.2,  175.3,   -2.5), (829., 1724.)],
        '4': [( 552.4, 1519.8,  -18.6), (499., 1438.)],
        '5': [( 215.7, 1720.9, 2868.1), (364., 274.)],
        '6': [( 214.3, 2028.1, 2862.0), (376., 320.)],
        '7': [(1425.8, 2030.8, 2856.1), (817., 324.)],
        '8': [(1421.9, 1726.0, 2861.8), (838., 278.)]
        },
    's006_xusen': {
        '1': [(  33.9,  174.0,   -0.5), (149., 1730.)],
        '2': [( 560.5,   83.7,  -11.1), (490., 1765.)],
        '3': [(1084.2,  175.3,   -2.5), (829., 1724.)],
        '4': [( 552.4, 1519.8,  -18.6), (499., 1437.)],
        '5': [( 215.7, 1720.9, 2868.1), (365., 273.)],
        '6': [( 214.3, 2028.1, 2862.0), (376., 320.)],
        '7': [(1425.8, 2030.8, 2856.1), (818., 322.)],
        '8': [(1421.9, 1726.0, 2861.8), (837., 277.)]
        },
    's007_zuogangao': {
        '1': [(  33.9,  174.0,   -0.5), (137., 1732.)],
        '2': [( 560.5,   83.7,  -11.1), (478., 1765.)],
        '3': [(1084.2,  175.3,   -2.5), (817., 1722.)],
        '4': [( 552.4, 1519.8,  -18.6), (487., 1438.)],
        '5': [( 215.7, 1720.9, 2868.1), (353., 273.)],
        '6': [( 214.3, 2028.1, 2862.0), (365., 319.)],
        '7': [(1425.8, 2030.8, 2856.1), (805., 325.)],
        '8': [(1421.9, 1726.0, 2861.8), (828., 280.)]
        },
    's008_liyu': {
        '1': [(  33.9,  174.0,   -0.5), (140., 1727.)],
        '2': [( 560.5,   83.7,  -11.1), (479., 1761.)],
        '3': [(1084.2,  175.3,   -2.5), (818., 1719.)],
        '4': [( 552.4, 1519.8,  -18.6), (489., 1434.)],
        '5': [( 215.7, 1720.9, 2868.1), (353., 272.)],
        '6': [( 214.3, 2028.1, 2862.0), (364., 317.)],
        '7': [(1425.8, 2030.8, 2856.1), (806., 321.)],
        '8': [(1421.9, 1726.0, 2861.8), (828., 277.)]
        },
    's009_sunyubo': {
        '1': [(  33.9,  174.0,   -0.5), (150., 1730.)],
        '2': [( 560.5,   83.7,  -11.1), (489., 1763.)],
        '3': [(1084.2,  175.3,   -2.5), (830., 1723.)],
        '4': [( 552.4, 1519.8,  -18.6), (498., 1433.)],
        '5': [( 215.7, 1720.9, 2868.1), (361., 273.)],
        '6': [( 214.3, 2028.1, 2862.0), (373., 319.)],
        '7': [(1425.8, 2030.8, 2856.1), (817., 323.)],
        '8': [(1421.9, 1726.0, 2861.8), (838., 277.)]
        },
    's010_handai': {
        '1': [(  33.9,  174.0,   -0.5), (153., 1733.)],
        '2': [( 560.5,   83.7,  -11.1), (494., 1767.)],
        '3': [(1084.2,  175.3,   -2.5), (835., 1727.)],
        '4': [( 552.4, 1519.8,  -18.6), (504., 1439.)],
        '5': [( 215.7, 1720.9, 2868.1), (370., 276.)],
        '6': [( 214.3, 2028.1, 2862.0), (382., 321.)],
        '7': [(1425.8, 2030.8, 2856.1), (823., 325.)],
        '8': [(1421.9, 1726.0, 2861.8), (844., 279.)]
        },
    's011_wuxingze': {
        '1': [(  33.9,  174.0,   -0.5), (151., 1734.)],
        '2': [( 560.5,   83.7,  -11.1), (491., 1767.)],
        '3': [(1084.2,  175.3,   -2.5), (831., 1728.)],
        '4': [( 552.4, 1519.8,  -18.6), (501., 1441.)],
        '5': [( 215.7, 1720.9, 2868.1), (367., 279.)],
        '6': [( 214.3, 2028.1, 2862.0), (377., 325.)],
        '7': [(1425.8, 2030.8, 2856.1), (819., 328.)],
        '8': [(1421.9, 1726.0, 2861.8), (840., 282.)]
        },
    's012_likaixiang': {
        },
    's013_zhangxiaohan': {
        },
    's014_maqichao': {
        },
    's015_weihuan': {
        },
    's017_tantian': {
        },
    's018_wangmian': {
        },
    's019_chenhongyuan': {
        },
    's020_houjikang': {
        }
    }

# TREADMILL_MAG_FIELD = np.array([-0.116, -0.320, -0.663])        # from one xsens
TREADMILL_MAG_FIELD = np.array([0.042, -0.529, -1.45])        # from another xsens
TREADMILL_MAG_FIELD = TREADMILL_MAG_FIELD / np.linalg.norm(TREADMILL_MAG_FIELD)

ALPHAPOSE_DETECTOR = os.environ.get('ALPHAPOSE_DETECTOR')








