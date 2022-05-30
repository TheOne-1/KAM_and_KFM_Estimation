import h5py
from const import DATA_PATH, SUBJECTS, FORCE_PHASE, R_FOOT_ORIENTATION_ERROR
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse


def get_ground_truth_fpa_list(data_sub, data_fields):
    fpa_true_list = []
    r_toe_loc = [data_fields.index(term) for term in ['RFM2_X', 'RFM2_Y', 'RFM2_Z']]
    r_heel_loc = [data_fields.index(term) for term in ['RFCC_X', 'RFCC_Y', 'RFCC_Z']]
    forward_vector = data_sub[:, :, r_toe_loc] - data_sub[:, :, r_heel_loc]
    force_phase = data_sub[:, :, data_fields.index(FORCE_PHASE)]
    for i_step in range(len(force_phase)):
        fpa_all_sample = 180 / np.pi * np.arctan2(forward_vector[i_step, :, 0], forward_vector[i_step, :, 1])
        stance_phase = np.where(force_phase[i_step] == 1)[0]
        strike, toeoff = stance_phase[0], stance_phase[-1]
        sample_15_gait_phase = int(round(strike + 0.2 * (toeoff - strike)))
        sample_50_gait_phase = int(round(strike + 0.8 * (toeoff - strike)))

        # plt.figure()
        # plt.plot(fpa_all_sample)
        # plt.plot([sample_15_gait_phase], [fpa_all_sample[sample_15_gait_phase]], '*')
        # plt.plot([sample_50_gait_phase], [fpa_all_sample[sample_50_gait_phase]], '*')
        # plt.show()
        fpa_true_list.append(np.mean(fpa_all_sample[sample_15_gait_phase:sample_50_gait_phase]))
    return fpa_true_list


def get_predicted_fpa_list(data_sub, data_fields, orientation_error):
    fpa_imu_list = list(data_sub[:, 0, data_fields.index('fpa_imu')])
    fpa_imu_list = [x - orientation_error for x in fpa_imu_list]
    return fpa_imu_list


def get_ground_truth_ts_list(data_sub, data_fields):
    fpa_true_list = []
    lsis_loc = [data_fields.index(term) for term in ['LIPS_X', 'LIPS_Y', 'LIPS_Z']]
    rsis_loc = [data_fields.index(term) for term in ['RIPS_X', 'RIPS_Y', 'RIPS_Z']]
    c7_loc = [data_fields.index(term) for term in ['CV7_X', 'CV7_Y', 'CV7_Z']]
    upward_vector = data_sub[:, :, c7_loc] - (data_sub[:, :, lsis_loc] + data_sub[:, :, rsis_loc]) / 2
    force_phase = data_sub[:, :, data_fields.index(FORCE_PHASE)]
    for i_step in range(len(force_phase)):
        ts_all_sample = 180 / np.pi * np.arctan2(upward_vector[i_step, :, 0], upward_vector[i_step, :, 2])
        fpa_true_list.append(np.max(ts_all_sample) - np.min(ts_all_sample))
    return fpa_true_list


def get_predicted_ts_list(data_sub, data_fields):
    fpa_imu_list = list(data_sub[:, 0, data_fields.index('step_trunk_sway_angle')])
    return fpa_imu_list


def evaluate_fpa():
    maes = []
    for sub in subjects:
        fpa_imu_list = get_predicted_fpa_list(data_all_sub[sub], data_fields, R_FOOT_ORIENTATION_ERROR[sub])
        fpa_true_list = get_ground_truth_fpa_list(data_all_sub[sub], data_fields)
        mae = np.mean(np.abs(np.array(fpa_imu_list) - np.array(fpa_true_list)))
        maes.append(mae)
        plt.figure()
        plt.plot(fpa_imu_list, fpa_true_list, '.')
        plt.plot([-20, 40], [-20, 40], 'black')
        ax = plt.gca()
        ax.set_xlim([-20, 40])
        ax.set_ylim([-20, 40])
    print("FPA: {:.1f} ± {:.1f}".format(np.mean(maes), np.std(maes)))


def evaluate_ts():
    maes = []
    for sub in subjects:
        ts_imu_list = get_predicted_ts_list(data_all_sub[sub], data_fields)
        ts_true_list = get_ground_truth_ts_list(data_all_sub[sub], data_fields)
        mae = np.mean(np.abs(np.array(ts_imu_list) - np.array(ts_true_list)))
        maes.append(mae)
        plt.figure()
        plt.plot(ts_imu_list, ts_true_list, '.')
    print("Trunk sway: {:.1f} ± {:.1f}".format(np.mean(maes), np.std(maes)))


with h5py.File(DATA_PATH + '/40samples+stance.h5', 'r') as hf:
    subjects = SUBJECTS
    data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items() if subject in subjects}
    data_fields = json.loads(hf.attrs['columns'])
    evaluate_fpa()
    evaluate_ts()

    # plt.show()






