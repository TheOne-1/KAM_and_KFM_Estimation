import h5py
from alan_framework import FourSourceModel, TianRNN
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

# define consts
JOINTS_90_FIELDS = [loc + axis + '_90' for loc in ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
                    for axis in ['_x', '_y']]
JOINTS_180_FIELDS = [loc + axis + '_180' for loc in ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
                     for axis in ['_x', '_y']]
IMU_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
ACC_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[:3]]
GYR_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[3:]]
R_FOOT_SHANK_GYR = ["Gyro" + axis + sensor for sensor in ['R_SHANK', 'R_FOOT'] for axis in ['X_', 'Y_', 'Z_']]

SEGMENT_MASS_PERCENT = {'L_FOOT': 1.37, 'R_FOOT': 1.37, 'R_SHANK': 4.33, 'R_THIGH': 14.16,
                        'WAIST': 11.17, 'CHEST': 15.96, 'L_SHANK': 4.33, 'L_THIGH': 14.16}


# define preprocess functions
def make_joints_relative_to_midhip():
    midhip_col_loc = [data_fields.index('MidHip' + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
    key_points_to_process = ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
    for sub_name, sub_data in data_all_sub.items():
        midhip_90_and_180_data = sub_data[:, :, midhip_col_loc]
        for key_point in key_points_to_process:
            key_point_col_loc = [data_fields.index(key_point + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
            sub_data[:, :, key_point_col_loc] = sub_data[:, :, key_point_col_loc] - midhip_90_and_180_data
            print(key_point_col_loc)
        data_all_sub[sub_name] = sub_data


def normalize_array_separately(data, scalar, method, scalar_mode='by_each_column'):
    input_data = data.copy()
    original_shape = input_data.shape
    target_shape = [-1, input_data.shape[2]] if scalar_mode == 'by_each_column' else [-1, 1]
    input_data[(input_data == 0.).all(axis=2), :] = np.nan
    input_data = input_data.reshape(target_shape)
    scaled_data = getattr(scalar, method)(input_data)
    scaled_data = scaled_data.reshape(original_shape)
    scaled_data[np.isnan(scaled_data)] = 0.
    return scaled_data


def get_body_weighted_imu():
    weight_col_loc = data_fields.index('body weight')
    for sub_name, sub_data in data_all_sub.items():
        sub_weight = sub_data[0, 0, weight_col_loc]
        for segment in ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']:
            segment_imu_col_loc = [data_fields.index(field + '_' + segment) for field in IMU_FIELDS[:6]]
            sub_data[:, :, segment_imu_col_loc[:3]] = \
                sub_data[:, :, segment_imu_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT[segment] / 100
        data_all_sub[sub_name] = sub_data


if __name__ == "__main__":
    """ step 0: select model and load data """
    # Three models are available: fusion, IMU_based, and camera_based
    model_name = 'IMU_based'
    # Two target moments: KAM or KFM
    target_moment = 'KAM'

    assert model_name in ['fusion', 'IMU_based', 'camera_based'], 'Incorrect model name.'
    assert target_moment in ['KAM', 'KFM'], 'Incorrect target moment name.'

    # one example data file is available
    with h5py.File('trained_models_and_example_data/example_data.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    model_path = './trained_models_and_example_data/' + model_name + '_' + target_moment + '.pth'
    model = torch.load(model_path)

    """ step 1: prepare subject 01's data as input """
    make_joints_relative_to_midhip()
    get_body_weighted_imu()
    subject_data = data_all_sub['subject_01']
    model_inputs = {}
    model_inputs['anthro'] = torch.from_numpy(subject_data[:, :, [data_fields.index('body weight'),
                                                                  data_fields.index('body height')]])
    model_inputs['step_length'] = torch.from_numpy(np.sum(~(subject_data[:, :, 0] == 0.), axis=1))

    # # switch axis when estimating KFM         # !!!
    # if target_moment == 'KFM':

    for submodel, component in zip([model.model_fx, model.model_fz, model.model_rx, model.model_rz],
                                   ['force_x', 'force_z', 'r_x', 'r_z']):
        input_fields = submodel.input_fields

        other_feature_loc = [data_fields.index(field) for field in input_fields if 'Acc' not in field]
        print([data_fields.index(field) for field in input_fields if '0' in field])
        subject_data[:, :, other_feature_loc] = normalize_array_separately(
            subject_data[:, :, other_feature_loc], model.scalars[component + '_other'], 'transform', scalar_mode='by_each_column')

        weighted_acc_loc = [data_fields.index(field) for field in input_fields if 'Acc' in field]
        if len(weighted_acc_loc) > 0:
            subject_data[:, :, weighted_acc_loc] = normalize_array_separately(
                subject_data[:, :, weighted_acc_loc], model.scalars[component + '_acc'], 'transform', scalar_mode='by_all_columns')
        submodel_input = subject_data[:, :, [data_fields.index(field) for field in input_fields]]
        model_inputs[component] = torch.from_numpy(submodel_input)

    """ step 2: predict KAM of subject 01 """
    predicted = model(model_inputs['force_x'], model_inputs['force_z'], model_inputs['r_x'], model_inputs['r_z'],
                      model_inputs['anthro'], model_inputs['step_length']).detach().numpy()

    """ step 3: plot estimation and true values """
    plt.figure()
    plt.plot(subject_data[:, :, data_fields.index('EXT_KM_Y')].ravel(), label='True Value')
    plt.plot(predicted.ravel(), label='Predicted Value')
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel('Time Step')
    ax.set_ylabel(target_moment + ' (BW X BH)')
    plt.title(model_name + ' model')
    plt.show()
