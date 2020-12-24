from vid_openpose.VideoReader import VideoReader
import os
from vid_openpose.const import SUB_NAMES, SUB_AND_TRIALS


dir_path = os.path.dirname(os.path.realpath(__file__))

for sub_name in SUB_NAMES[19:]:
    print(sub_name)
    for trial_name in SUB_AND_TRIALS[sub_name][3:]:
        print(trial_name)
        for camera in ['_90', '_180']:
            vid_dir = dir_path + '/../../../experiment_data/video/' + sub_name + '/' + trial_name + camera + '.MOV'
            model_dir = 'E:/ProgramFiles/vid_openpose-staf/STAF-staf/models/'
            output_dir = dir_path + '/../../../experiment_data/KAM/' + sub_name + '/video_output/' + trial_name + camera + '.csv'
            reader = VideoReader(vid_dir, model_dir, output_dir, True)
