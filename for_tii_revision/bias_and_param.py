import os
import h5py
import json
from figures.PaperFigures import get_mean_std, format_axis
from const import SUBJECTS
import numpy as np
import pandas as pd
from const import DATA_PATH
import matplotlib.pyplot as plt


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    sign = np.sign(v1_u[0] - v2_u[0])
    return sign * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def save_summary():
    with h5py.File('../figures/results/1108/TfnNet/results.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    summary = pd.DataFrame()
    subject_info = pd.read_csv(DATA_PATH + '/subject_info.csv', index_col=0)
    subject_info = subject_info.loc[SUBJECTS]
    for subject in SUBJECTS:
        data_sub = data_all_sub[subject]
        kam_mean_std = get_mean_std(data_sub, data_fields, 'EXT_KM_Y')
        kfm_mean_std = get_mean_std(data_sub, data_fields, 'EXT_KM_X')

        sub_bias_kam = np.mean(kam_mean_std['pred_mean'] - kam_mean_std['true_mean']) / np.mean(kam_mean_std['true_mean']) * 100
        sub_bias_kfm = np.mean(kfm_mean_std['pred_mean'] - kfm_mean_std['true_mean']) / np.mean(kfm_mean_std['true_mean']) * 100

        static_df = pd.read_csv(DATA_PATH + '/' + subject + '/combined/static_side.csv', index_col=0)
        static_df = static_df.mean()
        v_shank = (static_df['RKnee_x_90'] - static_df['RAnkle_x_90'], static_df['RKnee_y_90'] - static_df['RAnkle_y_90'])
        v_thigh = (static_df['RHip_x_90'] - static_df['RKnee_x_90'], static_df['RHip_y_90'] - static_df['RKnee_y_90'])

        knee_flexion_static = static_df['RIGHT_KNEE_FLEXION_ANGLE']
        knee_adduction_static = static_df['RIGHT_KNEE_ADDUCTION_ANGLE']

        shank_len = np.linalg.norm(v_shank)
        knee_angle_vid = np.rad2deg(angle_between(v_shank, v_thigh))
        sub_summary = pd.Series([sub_bias_kam, sub_bias_kfm, knee_flexion_static, knee_adduction_static, knee_angle_vid, shank_len], name=subject)
        summary = summary.append(sub_summary)

    summary.columns = ['sub_bias_kam', 'sub_bias_kfm', 'knee_flexion_static', 'knee_adduction_static', 'knee_angle_vid', 'shank_len']
    # summary['body height'] = subject_info['body height']
    # summary['body weight'] = subject_info['body weight']
    # summary['baseline speed'] = subject_info['baseline speed(m/s^2)']
    # summary['baseline_step_width'] = subject_info['baseline_step_width(m)']
    summary.to_csv(file_name)


if __name__ == "__main__":
    file_name = 'bias_summary.csv'
    # save_summary()

    data = pd.read_csv(file_name, index_col=0)
    for item in data.columns[2:]:
        plt.figure()
        ax = plt.gca()
        plt.scatter(data['sub_bias_kam'], data[item])
        for i, name in enumerate(SUBJECTS):
            ax.annotate(name, (data['sub_bias_kam'][i], data[item][i]))
        plt.xlabel('sub_bias_kam')
        plt.ylabel(item)

    # plt.figure()
    # plt.plot(data['knee_adduction_static'], data['knee_angle_vid'], '.')
    # plt.xlabel('knee_adduction_static')
    # plt.ylabel('knee_angle_vid')

    plt.show()




