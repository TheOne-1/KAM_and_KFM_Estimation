TRIALS = ['baseline', 'fpa', 'step_width', 'trunk_sway']
SUBJECTS = ['s001_tantian', 's002_wangdianxin', 's003_linyuan', 's004_ouyangjue', 's005_tangansheng', 's006_xusen',
            's007_zuogangao',
            's008_liyu', 's009_sunyubo', 's010_handai', 's011_wuxingze', 's012_likaixiang', 's013_zhangxiaohan',
            's014_maqichao',
            's015_weihuan', 's016_houjikang']
SENSOR_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
IMU_DATA_FIELDS = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS]

VIDEO_LIST = ["LShoulder", "RShoulder", "MidHip", "RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel",
              "LHeel"]
VIDEO_DATA_FIELDS = [VIDEO + "_" + position + "_" + angle for VIDEO in VIDEO_LIST for position in ["x", "y"] for
                     angle in ["90", "180"]]

SAMPLES_BEFORE_STEP = 40

TARGETS_LIST = ["RIGHT_KNEE_ADDUCTION_MOMENT"]
# all the fields of combined data
ALL_FIELDS = ["Event", "AccelX_L_FOOT", "AccelY_L_FOOT", "AccelZ_L_FOOT", "GyroX_L_FOOT", "GyroY_L_FOOT",
              "GyroZ_L_FOOT", "MagX_L_FOOT", "MagY_L_FOOT", "MagZ_L_FOOT", "Quat1_L_FOOT", "Quat2_L_FOOT",
              "Quat3_L_FOOT", "Quat4_L_FOOT", "AccelX_R_FOOT", "AccelY_R_FOOT", "AccelZ_R_FOOT", "GyroX_R_FOOT",
              "GyroY_R_FOOT", "GyroZ_R_FOOT", "MagX_R_FOOT", "MagY_R_FOOT", "MagZ_R_FOOT", "Quat1_R_FOOT",
              "Quat2_R_FOOT", "Quat3_R_FOOT", "Quat4_R_FOOT", "AccelX_R_SHANK", "AccelY_R_SHANK", "AccelZ_R_SHANK",
              "GyroX_R_SHANK", "GyroY_R_SHANK", "GyroZ_R_SHANK", "MagX_R_SHANK", "MagY_R_SHANK", "MagZ_R_SHANK",
              "Quat1_R_SHANK", "Quat2_R_SHANK", "Quat3_R_SHANK", "Quat4_R_SHANK", "AccelX_R_THIGH", "AccelY_R_THIGH",
              "AccelZ_R_THIGH", "GyroX_R_THIGH", "GyroY_R_THIGH", "GyroZ_R_THIGH", "MagX_R_THIGH", "MagY_R_THIGH",
              "MagZ_R_THIGH", "Quat1_R_THIGH", "Quat2_R_THIGH", "Quat3_R_THIGH", "Quat4_R_THIGH", "AccelX_WAIST",
              "AccelY_WAIST", "AccelZ_WAIST", "GyroX_WAIST", "GyroY_WAIST", "GyroZ_WAIST", "MagX_WAIST", "MagY_WAIST",
              "MagZ_WAIST", "Quat1_WAIST", "Quat2_WAIST", "Quat3_WAIST", "Quat4_WAIST", "AccelX_CHEST", "AccelY_CHEST",
              "AccelZ_CHEST", "GyroX_CHEST", "GyroY_CHEST", "GyroZ_CHEST", "MagX_CHEST", "MagY_CHEST", "MagZ_CHEST",
              "Quat1_CHEST", "Quat2_CHEST", "Quat3_CHEST", "Quat4_CHEST", "AccelX_L_SHANK", "AccelY_L_SHANK",
              "AccelZ_L_SHANK", "GyroX_L_SHANK", "GyroY_L_SHANK", "GyroZ_L_SHANK", "MagX_L_SHANK", "MagY_L_SHANK",
              "MagZ_L_SHANK", "Quat1_L_SHANK", "Quat2_L_SHANK", "Quat3_L_SHANK", "Quat4_L_SHANK", "AccelX_L_THIGH",
              "AccelY_L_THIGH", "AccelZ_L_THIGH", "GyroX_L_THIGH", "GyroY_L_THIGH", "GyroZ_L_THIGH", "MagX_L_THIGH",
              "MagY_L_THIGH", "MagZ_L_THIGH", "Quat1_L_THIGH", "Quat2_L_THIGH", "Quat3_L_THIGH", "Quat4_L_THIGH",
              "Nose_x_90", "Nose_y_90", "Nose_probability_90", "Neck_x_90", "Neck_y_90", "Neck_probability_90",
              "RShoulder_x_90", "RShoulder_y_90", "RShoulder_probability_90", "RElbow_x_90", "RElbow_y_90",
              "RElbow_probability_90", "RWrist_x_90", "RWrist_y_90", "RWrist_probability_90", "LShoulder_x_90",
              "LShoulder_y_90", "LShoulder_probability_90", "LElbow_x_90", "LElbow_y_90", "LElbow_probability_90",
              "LWrist_x_90", "LWrist_y_90", "LWrist_probability_90", "MidHip_x_90", "MidHip_y_90",
              "MidHip_probability_90", "RHip_x_90", "RHip_y_90", "RHip_probability_90", "RKnee_x_90", "RKnee_y_90",
              "RKnee_probability_90", "RAnkle_x_90", "RAnkle_y_90", "RAnkle_probability_90", "LHip_x_90", "LHip_y_90",
              "LHip_probability_90", "LKnee_x_90", "LKnee_y_90", "LKnee_probability_90", "LAnkle_x_90", "LAnkle_y_90",
              "LAnkle_probability_90", "REye_x_90", "REye_y_90", "REye_probability_90", "LEye_x_90", "LEye_y_90",
              "LEye_probability_90", "REar_x_90", "REar_y_90", "REar_probability_90", "LEar_x_90", "LEar_y_90",
              "LEar_probability_90", "LBigToe_x_90", "LBigToe_y_90", "LBigToe_probability_90", "LSmallToe_x_90",
              "LSmallToe_y_90", "LSmallToe_probability_90", "LHeel_x_90", "LHeel_y_90", "LHeel_probability_90",
              "RBigToe_x_90", "RBigToe_y_90", "RBigToe_probability_90", "RSmallToe_x_90", "RSmallToe_y_90",
              "RSmallToe_probability_90", "RHeel_x_90", "RHeel_y_90", "RHeel_probability_90", "Nose_x_180",
              "Nose_y_180", "Nose_probability_180", "Neck_x_180", "Neck_y_180", "Neck_probability_180",
              "RShoulder_x_180", "RShoulder_y_180", "RShoulder_probability_180", "RElbow_x_180", "RElbow_y_180",
              "RElbow_probability_180", "RWrist_x_180", "RWrist_y_180", "RWrist_probability_180", "LShoulder_x_180",
              "LShoulder_y_180", "LShoulder_probability_180", "LElbow_x_180", "LElbow_y_180", "LElbow_probability_180",
              "LWrist_x_180", "LWrist_y_180", "LWrist_probability_180", "MidHip_x_180", "MidHip_y_180",
              "MidHip_probability_180", "RHip_x_180", "RHip_y_180", "RHip_probability_180", "RKnee_x_180",
              "RKnee_y_180", "RKnee_probability_180", "RAnkle_x_180", "RAnkle_y_180", "RAnkle_probability_180",
              "LHip_x_180", "LHip_y_180", "LHip_probability_180", "LKnee_x_180", "LKnee_y_180", "LKnee_probability_180",
              "LAnkle_x_180", "LAnkle_y_180", "LAnkle_probability_180", "REye_x_180", "REye_y_180",
              "REye_probability_180", "LEye_x_180", "LEye_y_180", "LEye_probability_180", "REar_x_180", "REar_y_180",
              "REar_probability_180", "LEar_x_180", "LEar_y_180", "LEar_probability_180", "LBigToe_x_180",
              "LBigToe_y_180", "LBigToe_probability_180", "LSmallToe_x_180", "LSmallToe_y_180",
              "LSmallToe_probability_180", "LHeel_x_180", "LHeel_y_180", "LHeel_probability_180", "RBigToe_x_180",
              "RBigToe_y_180", "RBigToe_probability_180", "RSmallToe_x_180", "RSmallToe_y_180",
              "RSmallToe_probability_180", "RHeel_x_180", "RHeel_y_180", "RHeel_probability_180", "True_Event",
              "RIGHT_KNEE_ADDUCTION_MOMENT", "RIGHT_KNEE_FLEXION_MOMENT", "RIGHT_KNEE_ADDUCTION_ANGLE",
              "RIGHT_KNEE__ADDUCTION_VELOCITY", "LFCC_X", "LFCC_Y", "LFCC_Z", "LFM5_X", "LFM5_Y", "LFM5_Z", "LFM2_X",
              "LFM2_Y", "LFM2_Z", "RFCC_X", "RFCC_Y", "RFCC_Z", "RFM5_X", "RFM5_Y", "RFM5_Z", "RFM2_X", "RFM2_Y",
              "RFM2_Z", "LTAM_X", "LTAM_Y", "LTAM_Z", "LFAL_X", "LFAL_Y", "LFAL_Z", "LSK_X", "LSK_Y", "LSK_Z", "LTT_X",
              "LTT_Y", "LTT_Z", "RTAM_X", "RTAM_Y", "RTAM_Z", "RFAL_X", "RFAL_Y", "RFAL_Z", "RSK_X", "RSK_Y", "RSK_Z",
              "RTT_X", "RTT_Y", "RTT_Z", "LFME_X", "LFME_Y", "LFME_Z", "LFLE_X", "LFLE_Y", "LFLE_Z", "LTH_X", "LTH_Y",
              "LTH_Z", "LFT_X", "LFT_Y", "LFT_Z", "RFME_X", "RFME_Y", "RFME_Z", "RFLE_X", "RFLE_Y", "RFLE_Z", "RTH_X",
              "RTH_Y", "RTH_Z", "RFT_X", "RFT_Y", "RFT_Z", "LIPS_X", "LIPS_Y", "LIPS_Z", "RIPS_X", "RIPS_Y", "RIPS_Z",
              "LIAS_X", "LIAS_Y", "LIAS_Z", "RIAS_X", "RIAS_Y", "RIAS_Z", "MAI_X", "MAI_Y", "MAI_Z", "SXS_X", "SXS_Y",
              "SXS_Z", "SJN_X", "SJN_Y", "SJN_Z", "CV7_X", "CV7_Y", "CV7_Z", "LAC_X", "LAC_Y", "LAC_Z", "RAC_X",
              "RAC_Y", "RAC_Z"]
