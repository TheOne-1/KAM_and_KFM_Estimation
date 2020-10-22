"""
wearalbe_toolkit.py

@Author: Dianxin

This package is used to preprocess data collected from walking trials.
Read v3d exported csv data, sage csv data, vicon exported csv data and openPose exported csv data.
Synchronize vicon data and sage data.

"""
import csv
import numpy as np
import math
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import linalg
import wearable_math


class VideoCsvReader:
    """
    read video exported csv file by openPose.
    """

    def __init__(self, file_path):
        self.data_frame = pd.read_csv(file_path)

    def get_column_position(self, marker_name):
        return self.data_frame[marker_name]

    def get_rshank_angle(self):
        ankle = self.data_frame[['RAnkle_x', 'RAnkle_y']]
        knee = self.data_frame[['RKnee_x', 'RKnee_y']]
        vector = knee.values - ankle.values
        return np.arctan2(-vector[:, 1], vector[:, 0])

    def fill_low_probability_data(self):
        columns_label = self.data_frame.columns.values.reshape([-1, 3]).tolist()
        for x, y, probability in columns_label:
            self.data_frame.loc[self.data_frame[probability] < 0.6, [x, y]] = np.nan
        self.data_frame = self.data_frame.interpolate(method='linear', axis=0)

    def crop(self, start_index):
        # keep index after start_index
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = range(self.data_frame.shape[0])


class Visual3dCsvReader:
    """
    read v3d export data. It should contain LEFT_KNEE_MOMENT,LEFT_KNEE_ANGLE etc.
    """
    TRUE_EVENT_INDEX = 0

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, delimiter='\t', header=1, skiprows=[2, 3, 4])
        self.data.fillna(0)
        self.data_frame = self.data[[
            'RIGHT_KNEE_MOMENT', 'RIGHT_KNEE_MOMENT.1', 'RIGHT_KNEE_ANGLE', 'RIGHT_KNEE_VELOCITY']].fillna(0)
        self.data_frame.columns = ['RIGHT_KNEE_ADDUCTION_MOMENT', 'RIGHT_KNEE_FLEXION_MOMENT',
                                   'RIGHT_KNEE_ADDUCTION_ANGLE', 'RIGHT_KNEE__ADDUCTION_VELOCITY']

    def create_step_id(self, step_type):
        [LOFF, LON, ROFF, RON] = [self.data[event].dropna().values.tolist() for event in ['LOFF', 'LON', 'ROFF', 'RON']]

        events_dict = {'ROFF': ROFF, 'RON': RON, 'LOFF': LOFF, 'LON': LON}
        # Filter events_dict
        for _, frames in events_dict.items():
            for i in range(len(frames) - 1, -1, -1):
                if abs(frames[i] - frames[i - 1]) < 10:
                    frames.pop(i)
        foot_events = translate_step_event_to_step_id(events_dict, step_type)

        self.data_frame.insert(0, 'True_Event', np.nan)
        for _, event in foot_events.iterrows():
            self.data_frame.loc[event[0]:event[1], 'True_Event'] = Visual3dCsvReader.TRUE_EVENT_INDEX
            Visual3dCsvReader.TRUE_EVENT_INDEX += + 1

    def crop(self, start_index):
        # keep index after start_index
        self.data = self.data.loc[start_index:]
        self.data.index = range(self.data.shape[0])
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = range(self.data_frame.shape[0])


[PARENT_TITLE, SAMPLE_RATE, TITLE, DIRECTION, UNITS, DATA] = range(6)


class ViconCsvReader:
    '''
    Read the csv files exported from vicon.
    The csv file should only contain Trajectories information
    '''

    # if static_trial is not none, it's used for filling missing data.
    def __init__(self, file_path, segment_defitions=None, static_trail=None):
        self.data, self.sample_rate = ViconCsvReader.reading(file_path)
        # create segment marker data
        self.segment_data = dict()
        if segment_defitions is None:
            segment_defitions = {}
        for segment, markers in segment_defitions.items():
            self.segment_data[segment] = pd.Series(dict([(marker, self.data[marker]) for marker in markers]))

        # used for filling missing marker data
        if static_trail is not None:
            calibrate_data, _ = ViconCsvReader.reading(static_trail)
            for segment, markers in segment_defitions.items():
                segment_data = pd.Series(dict([(marker, calibrate_data[marker]) for marker in markers]))
                self.fill_missing_marker(segment_data, self.segment_data[segment])

        if segment_defitions != {}:
            markers = [marker for markers in segment_defitions.values() for marker in markers]
            self.data_frame = pd.concat([self.data[marker] for marker in markers], axis=1)
            self.data_frame.columns = [marker + '_' + axis for marker in markers for axis in ['X', 'Y', 'Z']]

    @staticmethod
    def reading(file_path):
        data_collection = dict()
        sample_rate_collection = {}
        state = PARENT_TITLE
        with open(file_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if state == PARENT_TITLE:
                    parent_title = row[0]
                    state = SAMPLE_RATE
                elif state == SAMPLE_RATE:
                    sample_rate_collection[parent_title] = float(row[0])
                    state = TITLE
                elif state == TITLE:
                    titles = list()
                    for col in row[2:]:
                        if col != "":
                            if 'Trajectories' == parent_title:
                                subject_name, title = col.split(':')
                            else:
                                title = col
                        titles.append(title)
                    state = DIRECTION
                elif state == DIRECTION:
                    directions = [i for i in row[2:]]
                    data = [[] for _ in directions]
                    state = UNITS
                elif state == UNITS:
                    # TODO@Dianxin: Record the units! never do it.
                    state = DATA
                elif state == DATA:
                    if row == []:
                        state = PARENT_TITLE
                        for title in titles:
                            data_collection[title] = dict()
                        for i, direction in enumerate(directions):
                            data_collection[titles[i]][direction] = data[i]
                        for key, value in data_collection.items():
                            data_collection[key] = pd.DataFrame(value)
                        continue
                    for i, x in enumerate(row[2:]):
                        try:
                            data[i].append(float(x))
                        except ValueError:
                            data[i].append(np.nan)
        return data_collection, sample_rate_collection

    def get_angular_velocity_theta(self, segment):
        segment_data_series = self.segment_data[segment]
        sampling_rate = self.sample_rate['Trajectories']

        walking_data = pd.concat(segment_data_series.tolist(), axis=1).values
        data_len = walking_data.shape[0]
        marker_number = int(walking_data.shape[1] / 3)
        angular_velocity_theta = np.zeros([data_len])

        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        # vectiorize this for loop.
        for i_frame in range(data_len):
            if i_frame == 0:
                continue
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame, :].reshape([marker_number, 3])
            R_one_sample, _ = rigid_transform_3D(current_marker_matrix, next_marker_matrix)
            theta = np.math.acos((np.matrix.trace(R_one_sample) - 1) / 2)

            angular_velocity_theta[i_frame] = theta * sampling_rate / np.pi * 180
        return angular_velocity_theta

    def get_rshank_angle(self, direction):
        ankle = (self.data['RTAM'] + self.data['RFAL']) / 2
        knee = (self.data['RFME'] + self.data['RFLE']) / 2
        vector = (knee - ankle).values
        if direction == 'X':
            return np.arctan2(vector[:, 2], vector[:, 1])
        elif direction == 'Y':
            return np.arctan2(vector[:, 2], vector[:, 0])
        elif direction == 'Z':
            return np.arctan2(vector[:, 1], vector[:, 0])

    def get_angular_velocity(self, segment, direction):
        segment_data_series = self.segment_data[segment]
        sampling_rate = self.sample_rate['Trajectories']
        walking_data = pd.concat(segment_data_series.tolist(), axis=1).values
        data_len = walking_data.shape[0]
        marker_number = int(walking_data.shape[1] / 3)
        angular_velocity = np.zeros([data_len, 3])

        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        if direction == 'X':
            next_marker_matrix[:, 0] = 0
        elif direction == 'Y':
            next_marker_matrix[:, 1] = 0
        elif direction == 'Z':
            next_marker_matrix[:, 2] = 0
        # vectiorize this for loop.
        for i_frame in range(1, data_len):
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame, :].reshape([marker_number, 3])
            if direction == 'X':
                next_marker_matrix[:, 0] = 0
            elif direction == 'Y':
                next_marker_matrix[:, 1] = 0
            elif direction == 'Z':
                next_marker_matrix[:, 2] = 0
            R_one_sample, _ = rigid_transform_3D(current_marker_matrix, next_marker_matrix)
            theta = rotationMatrixToEulerAngles(R_one_sample)

            angular_velocity[i_frame, :] = theta * sampling_rate / np.pi * 180
        angular_velocity = pd.DataFrame(angular_velocity)
        angular_velocity.columns = ['X', 'Y', 'Z']
        return angular_velocity[direction]

    def get_marker_position(self, marker_name):
        return self.data[marker_name]

    def crop(self, start_index):
        # keep index after start_index
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = range(self.data_frame.shape[0])

    def fill_missing_marker(self, calibrate_makers, motion_markers):
        if sum([motion_marker.isnull().sum().sum() for motion_marker in motion_markers.tolist()]) == 0:
            return

        # take the first frame of calibrate marker data for calibration
        calibrate_makers = pd.concat(calibrate_makers.tolist(), axis=1).values
        calibrate_makers = calibrate_makers[0, :].reshape([-1, 3])

        walking_data = pd.concat(motion_markers.tolist(), axis=1).values
        data_len = walking_data.shape[0]

        for i_frame in range(data_len):
            marker_matrix = walking_data[i_frame, :].reshape([-1, 3])
            coordinate_points = np.argwhere(~np.isnan(marker_matrix[:, 0])).reshape(-1)
            missing_points = np.argwhere(np.isnan(marker_matrix[:, 0])).reshape(-1)
            if len(missing_points) == 0:  # All the marker exist
                continue
            if len(coordinate_points) < 3:
                print("segment {} fail to filling missing marker with rigid body interpolation. Will use linear interpolation instead".format(motion_markers.index))
            else:
                origin, x, y, z = wearable_math.generat_coordinate(calibrate_makers[coordinate_points, :])
                origin_m, x_m, y_m, z_m = wearable_math.generat_coordinate(marker_matrix[coordinate_points, :])
                for missing_point in missing_points.tolist():
                    relative_point = wearable_math.get_relative_position(origin, x, y, z,
                                                                         calibrate_makers[missing_point, :])
                    missing_data = wearable_math.get_world_position(origin_m, x_m, y_m, z_m, relative_point)
                    motion_markers[missing_point]['X'][i_frame] = missing_data[0]
                    motion_markers[missing_point]['Y'][i_frame] = missing_data[1]
                    motion_markers[missing_point]['Z'][i_frame] = missing_data[2]
        for motion_marker in motion_markers:
            motion_marker.interpolate(method='linear', axis=0, inplace=True)



IMU_DATA_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Quat1', 'Quat2', 'Quat3', 'Quat4'] # We don't use these fields as they are noisy and position dependent.

class SageCsvReader:
    '''
    Read the csv file exported from sage systems
    '''
    GUESSED_EVENT_INDEX = 0
    sensor_list = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data_frame = self.data[
            [field + '_' + str(index) for index, label in enumerate(self.sensor_list) for field in
             IMU_DATA_FIELDS]].copy()
        index = self.data['Package_0']
        for i in range(1, len(self.data['Package_0'])):
            if self.data['Package_0'].loc[i] < self.data['Package_0'].loc[i - 1]:
                print("Warning sage imu data package index cross boundary from {} to {}".format(
                    self.data['Package_0'].loc[i],
                    self.data['Package_0'].loc[i - 1]))
                self.data.loc[i:, 'Package_0'] += 65536
        index = index - self.data['Package_0'].loc[0]
        # fill dropout data with nan
        self.data_frame.index = index
        self.data_frame.reindex(range(0, int(index.iloc[-1] + 1)))
        self.data_frame.columns = ["_".join([col.split('_')[0], self.sensor_list[int(col.split('_')[1])]]) for col in
                                   self.data_frame.columns]

    def get_norm(self, sensor, field, is_plot=False):
        assert sensor in self.sensor_list
        assert field in ['Accel', 'Gyro']
        norm_array = np.linalg.norm(self.data_frame[[field + direct + '_' + sensor for direct in ['X', 'Y', 'Z']]],
                                    axis=1)
        if is_plot:
            plt.figure()
            plt.plot(norm_array)
            plt.show()
        return norm_array

    def get_first_event_index(self):
        for i in range(len(self.data['sync_event'])):
            if self.data['sync_event'].loc[i] == 1:
                return i
        return None

    def get_field_data(self, sensor, field):
        if sensor not in self.sensor_list:
            raise RuntimeError("No such a sensor")
        if field not in ['Accel', 'Gyro']:
            raise RuntimeError("{field} not in ['Accel', 'Gyro']")
        index = str(self.sensor_list.index(sensor))
        data = self.data_frame[[field + direct + '_' + str(index) for direct in ['X', 'Y', 'Z']]]
        data.columns = ['X', 'Y', 'Z']
        return data

    def crop(self, start_index):
        self.data = self.data.loc[start_index:]
        self.data.index = self.data.index - self.data.index[0]
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = self.data_frame.index - self.data_frame.index[0]

    def create_step_id(self, step_type, verbose=False):
        # These highly impact the result
        flat_height_MAX_threshold = 100
        peaks_height = 100  # Optional Configuration 100
        peaks_distance = 30  # Optional Configuration 140

        [ROFF, RON, LOFF, LON] = [[] for _ in range(4)]
        l_foot_gyroX = self.data_frame['GyroX_L_FOOT']
        r_foot_gyroX = self.data_frame['GyroX_R_FOOT']

        for [foot_gyroX, ON, OFF] in [[l_foot_gyroX, LON, LOFF], [r_foot_gyroX, RON, ROFF]]:
            # detect strike and of
            peaks, _ = find_peaks(foot_gyroX, prominence=[10, None], height=peaks_height, distance=peaks_distance)
            # detect foot_flat
            is_last_foot_flat = True
            for i in range(1, len(peaks)):
                left = peaks[i - 1] + int(0.3 * (peaks[i] - peaks[i - 1]))
                right = peaks[i] - int(0.3 * (peaks[i] - peaks[i - 1]))
                is_foot_flat = abs(foot_gyroX[left:right].min()) < flat_height_MAX_threshold
                if not is_last_foot_flat and is_foot_flat:
                    OFF.append(peaks[i - 1])
                    ON.append(peaks[i])
                # used for debugging. Visualize the unexpected result. Then we can configure the parameters to gain better results.
                if i > 5 and is_last_foot_flat == is_foot_flat:
                    if verbose:
                        plt.figure()
                        plt.plot(range(peaks[i - 4], peaks[i]), foot_gyroX[peaks[i - 4]:peaks[i]])
                        plt.plot(peaks[i - 4:i], foot_gyroX[peaks[i - 4:i]], 'ro')
                        plt.show()

                    print('Warning: {} Foot Event.'.format('right' if ON == RON else 'left'))
                is_last_foot_flat = is_foot_flat

        events_dict = {'ROFF': ROFF, 'RON': RON, 'LOFF': LOFF, 'LON': LON}
        foot_events = translate_step_event_to_step_id(events_dict, step_type)
        self.data_frame.insert(0, 'Event', np.nan)
        for _, event in foot_events.iterrows():
            self.data_frame.loc[event[0]:event[1], 'Event'] = SageCsvReader.GUESSED_EVENT_INDEX
            SageCsvReader.GUESSED_EVENT_INDEX += 1


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rigid_transform_3D(A, B):
    '''
    Get the Rotation Matrix and Translation array between A and B.
    retval R: Rotation Matrix, 3*3
    retval T: Translation Array, 1*3
    '''
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(AA.T, BB)
    U, _, V_t = linalg.svd(H)
    R = np.dot(V_t.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        # print
        # "Reflection detected"
        V_t[2, :] *= -1
        R = np.dot(V_t.T, U.T)
    T = -np.dot(R, centroid_A.T) + centroid_B.T
    return R, T


def sync_via_correlation(data1, data2, verbose=False):
    correlation = np.correlate(data1, data2, 'full')
    delay = len(data2) - np.argmax(correlation) - 1
    if verbose:
        print(delay)
        plt.figure()
        if delay > 0:
            plt.plot(range(data1.size), data1)
            plt.plot(range(data2[delay:].size), data2[delay:])
        else:
            plt.plot(range(data1[-delay:].size), data1[-delay:])
            plt.plot(range(data2.size), data2)
        plt.show()
    return delay


def translate_step_event_to_step_id(events_dict, step_type):
    ## FILTER EVENTS
    r_steps = []
    if step_type == 'stance+swing':
        for i in range(4, len(events_dict['ROFF'])):
            if (events_dict['ROFF'][i] - events_dict['ROFF'][i - 1]) > 1.5 * (
                    events_dict['ROFF'][i - 1] - events_dict['ROFF'][i - 2]):
                continue
            r_steps.append([events_dict['ROFF'][i - 1], events_dict['ROFF'][i]])
    elif step_type == 'swing+stance':
        for i in range(4, len(events_dict['RON'])):
            if (events_dict['RON'][i] - events_dict['RON'][i - 1]) > 1.5 * (
                    events_dict['RON'][i - 1] - events_dict['RON'][i - 2]):
                continue
            r_steps.append([events_dict['RON'][i - 1], events_dict['RON'][i]])

    else:
        raise RuntimeError("no such step_type")

    right_events = pd.DataFrame(r_steps)
    right_events.columns = ['begin', 'end']
    return right_events


def calibrate_force_plate_center(vicon_data, plate_num):
    assert (plate_num in [1, 2])
    data_DL = vicon_data.data['DL']
    data_DR = vicon_data.data['DR']
    data_ML = vicon_data.data['ML']
    center_vicon = (data_DL + data_DR) / 2 + (data_DL - data_ML)
    if plate_num == 1:
        center_plate = vicon_data.data['Imported Bertec Force Plate #1 - CoP']
    else:
        center_plate = vicon_data.data['Imported Bertec Force Plate #2 - CoP']
    center_plate.columns = ['X', 'Y', 'Z']
    cop_offset = np.mean(center_plate, axis=0) - np.mean(center_vicon, axis=0)

    return cop_offset
