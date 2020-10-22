import unittest
import wearable_toolkit
import os

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

class MyTestCase(unittest.TestCase):
    def test_vicon_fill_missing_data(self):
        DATA_PATH = "../"
        subject = 's016_houjikang'
        vicon_motion_data_path = os.path.join(DATA_PATH, subject, 'test', 'static_back' + '.csv')
        vicon_static_data_path = os.path.join(DATA_PATH, subject, 'test', 'static_side' + '.csv')
        vicon_data = wearable_toolkit.ViconCsvReader(vicon_motion_data_path, SEGMENT_DEFITIONS, vicon_static_data_path)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
