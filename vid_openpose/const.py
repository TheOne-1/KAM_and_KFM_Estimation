import pyopenpose as op

TRIAL_NAMES = ['static_back', 'static_side', 'baseline', 'trunk_sway', 'fpa', 'step_width']

SUB_AND_TRIALS = {'s001_tantian': TRIAL_NAMES, 's002_wangdianxin': TRIAL_NAMES, 's003_linyuan': TRIAL_NAMES,
                  's004_ouyangjue': TRIAL_NAMES, 's005_tangansheng': TRIAL_NAMES, 's006_xusen': TRIAL_NAMES,
                  's007_zuogangao': TRIAL_NAMES, 's008_liyu': TRIAL_NAMES, 's009_sunyubo': TRIAL_NAMES,
                  's010_handai': TRIAL_NAMES, 's011_wuxingze': TRIAL_NAMES, 's012_likaixiang': TRIAL_NAMES,
                  's013_zhangxiaohan': TRIAL_NAMES, 's014_maqichao': TRIAL_NAMES, 's015_weihuan': TRIAL_NAMES,
                  's016_houjikang': TRIAL_NAMES, 's017_tantian': TRIAL_NAMES, 's018_wangmian': TRIAL_NAMES,
                  's019_chenhongyuan': TRIAL_NAMES, 's020_houjikang': TRIAL_NAMES}
SUB_NAMES = tuple(SUB_AND_TRIALS.keys())

# consts
KEY_POINT_DICT = op.getPoseBodyPartMapping(op.PoseModel.BODY_25)
KEY_POINT_DICT.pop(25)
"""    {
0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder',
6: 'LElbow', 7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee',
11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye',
16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe',
21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel'}
"""

KEY_POINT_COLUMN_NAME = [KEY_POINT_DICT[i_point] + '_' + axis for i_point in range(25)
                         for axis in ['x', 'y', 'probability']]
SEGMENT_POINT_MAP = {'r_thigh': [9, 10],  'l_thigh': [12, 13], 'r_shank': [10, 11], 'l_shank': [13, 14]}
VICON_SAMPLE_RATE = 100


