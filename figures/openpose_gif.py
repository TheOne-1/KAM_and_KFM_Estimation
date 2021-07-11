from VideoReader import VideoReader
from const import SUBJECTS, TRIALS, VIDEO_ORIGINAL_SAMPLE_RATE
import os
import cv2
import imageio
import pyopenpose as op
from PIL import Image

VIDEO_PATH = 'D:\Tian\Research\Projects\VideoIMUCombined\experiment_data\\video'
OPENPOSE_MODEL_PATH = 'E:\ProgramFiles\openpose-master\models'


class VideoToGif(VideoReader):
    def __init__(self, vid_dir, op_model_dir):
        self._vid_dir = vid_dir
        self._op_model_dir = op_model_dir
        self.key_points_df = self.create_gif(5000, 5250)

    def create_gif(self, start_frame, end_frame):

        """ Only to create some example figures """
        cap = cv2.VideoCapture(self._vid_dir)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if 0 > start_frame:
            raise ValueError('Only positive i frame.')
        elif end_frame >= self.frame_count:
            raise ValueError('i frame larger than the video length.')
        # Starting OpenPose
        op_params = dict()
        op_params['model_folder'] = self._op_model_dir
        op_params["model_pose"] = "BODY_25"
        op_params['number_people_max'] = 1
        opWrapper = op.WrapperPython()
        opWrapper.configure(op_params)
        opWrapper.start()
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        frames = []
        for i_frame in range(start_frame, end_frame, 4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            retrieve_flag, frame = cap.read()
            # key points will be all zero if the frame is not retrieved.
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            datum = op.Datum()
            datum.cvInputData = frame

            opWrapper.emplaceAndPop([datum])
            cv_output = datum.cvOutputData
            cv_output = cv2.cvtColor(cv_output, cv2.COLOR_BGR2RGB)
            cv_output = cv_output[int(vid_width*0.25):int(vid_width*0.82), int(vid_height*0.2):int(vid_height*0.75), :]
            cv_output = cv2.resize(cv_output, (int(vid_height / 4), int(vid_width / 4)))
            frames.append(cv_output)

        if '90' in self._vid_dir:
            imageio.mimsave('exports/gif_90.gif', frames, fps=30)
        else:
            imageio.mimsave('exports/gif_180.gif', frames, fps=30)



if __name__ == '__main__':
    VICON_SAMPLE_RATE = 100
    sub_name = SUBJECTS[-4]
    trial_name = TRIALS[0]
    for camera in ['_90', '_180']:
        video_file_name = os.path.join(VIDEO_PATH, sub_name, trial_name + camera + '.MOV')
        reader = VideoToGif(video_file_name, OPENPOSE_MODEL_PATH)
