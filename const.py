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