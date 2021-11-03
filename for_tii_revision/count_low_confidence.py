from const import SEGMENT_DEFINITIONS, SUBJECTS, STATIC_TRIALS, TRIALS, DATA_PATH, SUBJECT_HEIGHT, SUBJECT_WEIGHT
import wearable_toolkit
import pandas as pd
import numpy as np
import os

target_prob = [segment + '_probability' for segment in
               ['LShoulder', 'RShoulder', 'MidHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']]

high_prob_num, total_num, max_len = 0, 0, 0
for subject in SUBJECTS[:]:
    for trial in TRIALS:
        for video_angle in ['90', '180']:
            video_data_path = os.path.join(DATA_PATH, subject, 'raw_video_output', trial + '_' + video_angle + '.csv')
            video_data = wearable_toolkit.VideoCsvReader(video_data_path).data_frame
            prob = video_data[target_prob]
            high_prob = prob.gt(0.5)
            for prob_name in target_prob:
                high_prob_col = high_prob[prob_name]
                high_prob_num += high_prob_col.value_counts()[True]
                total_num += high_prob_col.shape[0]

                # temp = (high_prob[prob_name].diff(False) != 0).astype('int').cumsum()
                groups = (high_prob_col != high_prob_col.shift()).cumsum()
                groups_false = pd.DataFrame({'value_grp': groups[high_prob_col == False]})
                lens = pd.DataFrame({'Consecutive': groups_false.groupby('value_grp').size()}).reset_index(drop=True)
                if len(lens.values) > 0 and max_len < max(lens.values):
                    max_len = max(lens.values)[0]
print('{:3.3f}%, {}'.format(100 * (total_num - high_prob_num) / total_num, max_len))
