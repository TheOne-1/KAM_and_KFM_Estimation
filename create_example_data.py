import h5py
import os
import json
from const import SUBJECTS, ALL_FIELDS

EXAMPLE_DATA_FIELDS = [
    'body weight', 'body height', 'force_phase'

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

with h5py.File(os.environ.get('KAM_DATA_PATH') + '/40samples+stance.h5', 'r') as hf:
    data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
    data_fields = json.loads(hf.attrs['columns'])
    export_path = 'trained_models_and_example_data/example_data.h5'
    example_col_loc = [data_fields.index(column) for column in EXAMPLE_DATA_FIELDS]
    with h5py.File(export_path, 'w') as hf:
        hf.create_dataset('subject_01', data=data_all_sub[SUBJECTS[7]][:10, :, example_col_loc], dtype='float32')
        hf.create_dataset('subject_02', data=data_all_sub[SUBJECTS[8]][:10, :, example_col_loc], dtype='float32')
        hf.attrs['columns'] = json.dumps(data_fields)
