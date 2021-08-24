import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from const import SUBJECTS, DATA_PATH, TRIALS, TARGETS_LIST
from transforms3d.quaternions import rotate_vector, mat2quat, quat2mat
from triangulation.triangulation_toolkit import q_to_knee_angle, init_kalman_param, compare_axes_results
from triangulation.triangulation_toolkit import plot_q_for_debug


def get_orientation_from_vectors(segment_z, segment_ml):
    segment_y = np.cross(segment_z, segment_ml)
    segment_x = np.cross(segment_y, segment_z)
    fun_norm_vect = lambda v: v / np.linalg.norm(v)
    segment_x = np.apply_along_axis(fun_norm_vect, 1, segment_x)
    segment_y = np.apply_along_axis(fun_norm_vect, 1, segment_y)
    segment_z = np.apply_along_axis(fun_norm_vect, 1, segment_z)

    R_body_glob = np.array([segment_x, segment_y, segment_z])
    R_body_glob = np.swapaxes(R_body_glob, 0, 1)
    R_glob_body = np.swapaxes(R_body_glob, 1, 2)

    def temp_fun(R):
        if np.isnan(R).any():
            return np.array([1, 0, 0, 0])
        else:
            quat = mat2quat(R)
            if quat[3] < 0:
                quat = - quat
            return quat / np.linalg.norm(quat)

    q_glob_body = np.array(list(map(temp_fun, R_glob_body)))
    return q_glob_body


def angle_between_vectors(subject, trial):
    vid_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'triangulated', trial+'.csv'), index_col=0)
    joint_col = [joint + '_3d_' + axis for joint in ['RHip', 'RKnee', 'RAnkle'] for axis in ['x', 'y', 'z']]
    joint_hip, joint_knee, joint_ankle = [vid_data[joint_col[3*i:3*(i+1)]].values for i in range(3)]
    shank_y, shank_ml = joint_knee - joint_ankle, [1, 0, 0]
    q_shank_glob_body = get_orientation_from_vectors(shank_y, shank_ml)
    thigh_y, thigh_ml = joint_hip - joint_knee, [1, 0, 0]
    q_thigh_glob_body = get_orientation_from_vectors(thigh_y, thigh_ml)
    knee_angles_esti = q_to_knee_angle(q_shank_glob_body, q_thigh_glob_body, np.eye(3), np.eye(3))
    return knee_angles_esti, q_shank_glob_body, q_thigh_glob_body


for subject in SUBJECTS[:2]:
    for trial in TRIALS:
        knee_angles_esti, q_shank_glob_body, q_thigh_glob_body = angle_between_vectors(subject, trial)

        trial_static_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', 'static_back.csv'), index_col=0)
        knee_angles_vicon_static = trial_static_data[TARGETS_LIST[:3]].values

        trial_data = pd.read_csv(os.path.join(DATA_PATH, subject, 'combined', trial + '.csv'), index_col=False)
        knee_angles_vicon = trial_data[TARGETS_LIST[:3]].values

        knee_angles_vicon = knee_angles_vicon - np.mean(knee_angles_vicon_static, axis=0)  # to remove static knee angle
        compare_axes_results(knee_angles_vicon, knee_angles_esti, ['Flexion', 'Adduction', 'IE'],
                             start=0, end=trial_data.shape[0])
plt.show()


