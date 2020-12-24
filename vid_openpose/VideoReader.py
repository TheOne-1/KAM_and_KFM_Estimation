import cv2
import numpy as np
import pandas as pd
from numpy import mean, tile, dot, linalg
import scipy.interpolate as interpo
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, filtfilt
import pyopenpose as op
from a0_VideoProcess.const import KEY_POINT_DICT, VICON_SAMPLE_RATE, KEY_POINT_COLUMN_NAME, SEGMENT_POINT_MAP


class VideoReader:

    def __init__(self, vid_dir, op_model_dir, output_dir, show_op_result=False):
        self._vid_dir = vid_dir
        self._op_model_dir = op_model_dir
        self.show_op_result = show_op_result

        self.key_points_df = self._get_key_points()

        # compute and save the shank angular velocity
        # orientations = self.get_segment_2d_orientation('l_shank', self.key_points_df, self._vid_fps, 10)
        # angluar_vels = self.orientation_to_angluar_velocity_2d(orientations, self._vid_fps)
        # self.key_points_df.insert(self.key_points_df.shape[1], 'l_shank_angular_vel', angluar_vels)
        #
        # orientations = self.get_segment_2d_orientation('r_shank', self.key_points_df, self._vid_fps, 10)
        # angluar_vels = self.orientation_to_angluar_velocity_2d(orientations, self._vid_fps)
        # self.key_points_df.insert(self.key_points_df.shape[1], 'r_shank_angular_vel', angluar_vels)

        self.export_result(VICON_SAMPLE_RATE, output_dir)

    def _get_key_points(self, start_frame=0, end_frame=None):
        cap = cv2.VideoCapture(self._vid_dir)

        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vid_fps = cap.get(cv2.CAP_PROP_FPS)
        # if self._vid_file['start'] is None:
        #     start_frame = 0
        # else:
        #     start_frame = int(self._vid_file['start'] * self._vid_fps)
        #
        if end_frame is None:
            end_frame = self.frame_count

        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        key_points = np.zeros([end_frame - start_frame, 75])

        # Starting OpenPose
        op_params = dict()
        op_params['model_folder'] = self._op_model_dir
        op_params["model_pose"] = "BODY_21"
        op_params['number_people_max'] = 1
        opWrapper = op.WrapperPython()
        opWrapper.configure(op_params)
        opWrapper.start()

        completion_percent = 0     # 添加进度条
        last_phase_end_time, last_phase_end_frame = time.time(), 0
        for i_frame in range(end_frame - start_frame):
            # print the status
            completion_percent_new = np.floor((i_frame + 2) / (end_frame - start_frame) * 10) / 10
            if completion_percent_new != completion_percent:
                current_time = time.time()
                fps = round((i_frame - last_phase_end_frame) / (current_time - last_phase_end_time), 1)
                last_phase_end_time, last_phase_end_frame = current_time, i_frame
                completion_percent = completion_percent_new
                print(str(int(100 * completion_percent)) + '%  ' + str(fps) + ' fps', end='\t', flush=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame + start_frame)
            retval, frame = cap.read()
            # key points will be all zero if the frame is not retrieved.
            if not retval:
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
        key_points_df.columns = KEY_POINT_COLUMN_NAME
        return key_points_df

    @staticmethod
    def get_segment_2d_orientation(segment, key_points_df, vid_fps, cut_off_fre=None):
        """

        :param segment: str of the segment name, check SEGMENT_POINT_MAP for details
        SEGMENT_POINT_MAP = {'r_thigh': [9, 10],  'l_thigh': [12, 13], 'r_shank': [10, 11], 'l_shank': [13, 14]}
        :return:
        """
        if segment not in SEGMENT_POINT_MAP.keys():
            raise ValueError('Invalid segment name.')

        segment_point_ids = SEGMENT_POINT_MAP[segment]
        segment_point_names = [KEY_POINT_DICT[segment_point_id] for segment_point_id in segment_point_ids]
        column_names = [segment_point_name + '_' + axis for segment_point_name in segment_point_names for axis in ['x', 'y']]

        delta_x = key_points_df[column_names[0]] - key_points_df[column_names[2]]
        # flip the sign of y axis because of coordinate difference between vicon and iphone image
        delta_y = - (key_points_df[column_names[1]] - key_points_df[column_names[3]])
        orientations = np.arctan2(delta_y, delta_x).values
        if cut_off_fre is not None:
            orientations = VideoReader.data_filt(orientations, 10, vid_fps)
        return orientations

    @staticmethod
    def orientation_to_angluar_velocity_2d(orientations, vid_fps):
        data_len = orientations.shape[0]
        orientations = orientations.reshape([-1, 1])
        steps = np.arange(0, data_len / vid_fps - 1e-10, 1 / vid_fps)
        tck, step = interpo.splprep(orientations.T, u=steps, s=0)
        angluar_vels = interpo.splev(steps, tck, der=1)[0]
        return angluar_vels

    def export_result(self, target_sample_rate, output_dir):
        data_len = self.key_points_df.shape[0]

        ori_steps = np.arange(0, data_len / self._vid_fps - 1e-10, 1 / self._vid_fps)
        target_steps = np.arange(0, data_len / self._vid_fps - 1e-10, 1 / target_sample_rate)
        interpolated_key_points = np.zeros([target_steps.shape[0], self.key_points_df.shape[1]])

        # interpolate each key point
        for i_point in KEY_POINT_DICT.keys():
            point_name = KEY_POINT_DICT[i_point]
            point_column_names = [point_name + '_' + axis for axis in ['x', 'y', 'probability']]
            point_data = self.key_points_df[point_column_names].values
            tck, step = interpo.splprep(point_data.T, u=ori_steps, s=0)
            for i_axis in range(3):
                interpolated_key_points[:, i_point*3+i_axis] = interpo.splev(target_steps, tck, der=0)[i_axis]

        # interpolate rest columns
        for i_axis in range(len(KEY_POINT_DICT.keys())*3, len(self.key_points_df.columns)):
            angular_vel_data = self.key_points_df.iloc[:, i_axis].values.reshape([-1, 1])
            tck, step = interpo.splprep(angular_vel_data.T, u=ori_steps, s=0)
            interpolated_key_points[:, i_axis] = interpo.splev(target_steps, tck, der=0)[0]

        interpolated_key_point_df = pd.DataFrame(interpolated_key_points, columns=self.key_points_df.columns)
        interpolated_key_point_df.to_csv(output_dir, index=False)

        # plt.figure()
        # plt.plot(interpolated_key_point_df.iloc[:, -1])
        # plt.plot(self.key_points_df.iloc[:, -1])
        # plt.figure()
        # plt.plot(interpolated_key_point_df.iloc[:, 63])
        # plt.plot(self.key_points_df.iloc[:, 63])
        # plt.show()

    @staticmethod
    def data_filt(data, cut_off_fre, sampling_fre, filter_order=4):
        fre = cut_off_fre / (sampling_fre / 2)
        b, a = butter(filter_order, fre, 'lowpass')
        if len(data.shape) == 1:
            data_filt = filtfilt(b, a, data)
        else:
            data_filt = filtfilt(b, a, data, axis=0)
        return data_filt

    @staticmethod
    def rigid_transform(A, B):
        """
        Use this function when the segment is defined by three or more key points.

        segment_point_num = len(segment_point_ids)
        segment_data = self.key_points_df[column_names].values
        ori_matrix = segment_data[0, :].reshape([segment_point_num, 2])
        rotation_angles = np.zeros([self.frame_count])
        for i_frame in range(self.frame_count):
            current_point_matrix = segment_data[i_frame, :].reshape([segment_point_num, 2])
            rotation_angles[i_frame], t = self.rigid_transform(ori_matrix, current_point_matrix)

        :param A:
        :param B:
        :return:
        """

        assert len(A) == len(B)

        N = A.shape[0]  # total points
        centroid_A = mean(A, axis=0)
        centroid_B = mean(B, axis=0)
        # centre the points
        AA = A - tile(centroid_A, (N, 1))
        BB = B - tile(centroid_B, (N, 1))
        # dot is matrix multiplication for array
        H = dot(AA.T, BB)
        U, S, Vt = linalg.svd(H)
        R = dot(Vt.T, U.T)

        # special reflection case
        if linalg.det(R) < 0:
            # "Reflection detected"
            Vt[1, :] *= -1
            R = dot(Vt.T, U.T)
        t = -dot(R, centroid_A.T) + centroid_B.T
        angle = np.arccos(R[0, 0])
        return angle, t
