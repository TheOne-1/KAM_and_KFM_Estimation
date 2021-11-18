import os
import time
import datetime
import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import pyopenpose as op
from const import SUBJECTS, TRIALS, STATIC_TRIALS, DATA_PATH, VIDEO_PATH, OPENPOSE_MODEL_PATH, ALPHAPOSE_DETECTOR
import torch
import sys
# sys.path.insert(0, ALPHAPOSE_DETECTOR)
sys.path.append(os.path.dirname(ALPHAPOSE_DETECTOR))
from scripts.demo_api import SingleImageAlphaPose
# from detector.yolo_cfg import cfg
import argparse
from alphapose.utils.config import update_config


# consts
KEY_POINT_DICT_OP = op.getPoseBodyPartMapping(op.PoseModel.BODY_25)
KEY_POINT_DICT_OP.pop(25)
"""    {
0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder',
6: 'LElbow', 7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee',
11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye',
16: 'LEye', 17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe',
21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel'}
"""
KEY_POINT_COLUMN_NAME_OP = [KEY_POINT_DICT_OP[i_point] + '_' + axis for i_point in range(25)
                            for axis in ['x', 'y', 'probability']]


class VideoReader:
    def __init__(self, vid_dir, show_op_result):
        self._vid_dir = vid_dir
        self.show_op_result = show_op_result
        self.key_points_df = self._get_key_points()
        self.save_results()

    def _get_key_points(self):
        raise RuntimeError('VideoReader is an abstract class, Use VideoReaderOpenPose or VideoReaderAlphaPose instead')

    def save_results(self):
        raise RuntimeError('VideoReader is an abstract class, Use VideoReaderOpenPose or VideoReaderAlphaPose instead')

    @staticmethod
    def data_filt(data, cut_off_fre, sampling_fre, filter_order=4):
        fre = cut_off_fre / (sampling_fre / 2)
        b, a = butter(filter_order, fre, 'lowpass')
        if len(data.shape) == 1:
            data_filt = filtfilt(b, a, data)
        else:
            data_filt = filtfilt(b, a, data, axis=0)
        return data_filt


class VideoReaderOpenPose(VideoReader):
    def __init__(self, vid_dir, op_model_dir, show_op_result=False):
        self._op_model_dir = op_model_dir
        super().__init__(vid_dir, show_op_result)

    def _get_key_points(self, start_frame=0, end_frame=None):
        cap = cv2.VideoCapture(self._vid_dir)

        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if end_frame is None:
            end_frame = self.frame_count

        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        key_points = np.zeros([end_frame - start_frame, 75])

        # Starting OpenPose
        op_params = dict()
        op_params['model_folder'] = self._op_model_dir
        op_params["model_pose"] = "BODY_25"
        op_params['number_people_max'] = 1
        opWrapper = op.WrapperPython()
        opWrapper.configure(op_params)
        opWrapper.start()

        last_phase_end_time = time.time()
        for i_frame in range(start_frame, end_frame):
            # print the status
            video_time = i_frame // self._vid_fps
            current_time = time.time()
            fps = round(1 / (current_time - last_phase_end_time), 2)
            last_phase_end_time = current_time
            print("\rframe index:{} \tat time:{} \tat percent:{}% \tremaining time:{} \tfps:{}".format(
                i_frame, datetime.timedelta(seconds=video_time), round(i_frame / self.frame_count * 100, 1),
                datetime.timedelta(seconds=int((end_frame - i_frame)/ fps)), fps), end='')

            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame + start_frame)
            retrieve_flag, frame = cap.read()
            # key points will be all zero if the frame is not retrieved.
            if not retrieve_flag:
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            datum = op.Datum()
            datum.cvInputData = frame

            opWrapper.emplaceAndPop([datum])
            key_points[i_frame, :] = datum.poseKeypoints.ravel()

            if self.show_op_result:
                cv_output = cv2.resize(datum.cvOutputData, (int(vid_height / 2), int(vid_width / 2)))  # Resize image
                cv2.imshow('No title', cv_output)
                cv2.waitKey(10)
        print()
        cv2.destroyAllWindows()
        key_points_df = pd.DataFrame(key_points)
        key_points_df.columns = KEY_POINT_COLUMN_NAME_OP
        return key_points_df

    def save_results(self):
        output_file_name = os.path.join(DATA_PATH, sub_name, 'raw_video_output_openpose', trial_name + camera + '.csv')
        self.key_points_df.to_csv(output_file_name)


KEY_POINT_DICT_AP = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist',
                     'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
KEY_POINT_COLUMN_NAME_AP = [KEY_POINT_DICT_AP[i_point] + '_' + axis for i_point in range(17)
                            for axis in ['x', 'y', 'probability']]


class VideoReaderAlphaPose(VideoReader):
    def __init__(self, vid_dir, no_use, show_result=True):
        self.single_image_ap = self.load_single_image_ap()
        super().__init__(vid_dir, show_result)

    def load_single_image_ap(self):
        parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
        parser.add_argument('--cfg', type=str,
                            default=ALPHAPOSE_DETECTOR+'/../pretrained_models/Fast Pose (DCN).yaml')
        parser.add_argument('--checkpoint', type=str,
                            default=ALPHAPOSE_DETECTOR+'/../pretrained_models/Fast Pose (DCN).pth')
        parser.add_argument('--detector', dest='detector',
                            help='detector name', default="yolo")
        parser.add_argument('--image', dest='inputimg',
                            help='image-name', default="")
        parser.add_argument('--save_img', default=False, action='store_true',
                            help='save result as image')
        parser.add_argument('--vis', default=False, action='store_true',
                            help='visualize image')
        parser.add_argument('--showbox', default=False, action='store_true',
                            help='visualize human bbox')
        parser.add_argument('--profile', default=False, action='store_true',
                            help='add speed profiling at screen output')
        parser.add_argument('--format', type=str,
                            help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
        parser.add_argument('--min_box_area', type=int, default=0,
                            help='min box area to filter out')
        parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                            help='save the result json as coco format, using image index(int) instead of image name(str)')
        parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                            help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
        parser.add_argument('--flip', default=False, action='store_true',
                            help='enable flip testing')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='print detail information')
        parser.add_argument('--vis_fast', dest='vis_fast',
                            help='use fast rendering', action='store_true', default=False)
        """----------------------------- Tracking options -----------------------------"""
        parser.add_argument('--pose_flow', dest='pose_flow',
                            help='track humans in video with PoseFlow', action='store_true', default=False)
        parser.add_argument('--pose_track', dest='pose_track',
                            help='track humans in video with reid', action='store_true', default=False)
        args = parser.parse_args()
        cfg = update_config(args.cfg)
        args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

        return SingleImageAlphaPose(args, cfg)

    def _get_key_points(self, start_frame=0, end_frame=None):
        cap = cv2.VideoCapture(self._vid_dir)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if end_frame is None:
            end_frame = self.frame_count

        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        key_points = np.zeros([end_frame - start_frame, 51])
        last_phase_end_time = time.time() - 1e-5
        for i_frame in range(start_frame, end_frame):
            # print the status
            video_time = i_frame // self._vid_fps
            current_time = time.time()
            fps = round(1 / (current_time - last_phase_end_time), 2)
            last_phase_end_time = current_time
            print("\rframe index:{} \tat time:{} \tat percent:{}% \tremaining time:{} \tfps:{}".format(
                i_frame, datetime.timedelta(seconds=video_time), round(i_frame / self.frame_count * 100, 1),
                datetime.timedelta(seconds=int((end_frame - i_frame)/ fps)), fps), end='')

            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame + start_frame)
            retrieve_flag, frame = cap.read()
            # key points will be all zero if the frame is not retrieved.
            if not retrieve_flag:
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            pose = self.single_image_ap.process('', frame)
            result = pose['result'][0]
            key_points[i_frame, :] = np.column_stack([result['keypoints'].numpy(), result['kp_score'].numpy()]).ravel()
            if self.show_op_result:
                img = self.single_image_ap.vis(frame, pose)  # visulize the pose result
                cv_output = cv2.resize(img, (int(vid_height / 2), int(vid_width / 2)))  # Resize image
                cv2.imshow('No title', cv_output)
                cv2.waitKey(10)
        print()
        cv2.destroyAllWindows()
        key_points_df = pd.DataFrame(key_points)
        key_points_df.columns = KEY_POINT_COLUMN_NAME_AP
        return key_points_df

    def save_results(self):
        output_file_name = os.path.join(DATA_PATH, sub_name, 'raw_video_output_alphapose', trial_name + camera + '.csv')
        self.key_points_df.to_csv(output_file_name)


if __name__ == '__main__':
    for sub_name in SUBJECTS:
        print(sub_name)
        for trial_name in TRIALS+STATIC_TRIALS:
            print(trial_name)
            for camera in ['_90', '_180']:
                video_file_name = os.path.join(VIDEO_PATH, sub_name, trial_name + camera + '.MOV')
                reader = VideoReaderAlphaPose(video_file_name, False)
