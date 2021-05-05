import h5py
import copy
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
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
GRAVITY = 9.81


# define deep learning model
class TianRNN(nn.Module):
    def __init__(self, x_dim, y_dim, input_fields, seed=0, nlayer=2):
        super(TianRNN, self).__init__()
        torch.manual_seed(seed)
        self.rnn_layer = nn.LSTM(x_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.y_dim = y_dim
        self.r2d = nn.Linear(2 * globals()['lstm_unit'], globals()['fcnn_unit'], bias=False)
        self.d2o = nn.Linear(globals()['fcnn_unit'], y_dim, bias=False)
        self.relu = nn.ReLU()
        self.input_fields = input_fields
        for layer in [self.r2d, self.d2o]:
            nn.init.xavier_normal_(layer.weight)
        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sequence, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.rnn_layer(sequence)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=152)
        sequence = self.r2d(lstm_out)
        sequence = self.relu(sequence)
        output = self.d2o(sequence)
        return output


class FourSourceModel(nn.Module):
    def __init__(self, model_fx, model_fz, model_rx, model_rz, scalars):
        super(FourSourceModel, self).__init__()
        self.model_fx = model_fx
        self.model_fz = model_fz
        self.model_rx = model_rx
        self.model_rz = model_rz
        self.scalars = scalars

    def forward(self, x_fx, x_fz, x_rx, x_rz, anthro, lens):
        out_fx = self.model_fx(x_fx, lens)
        out_fz = self.model_fz(x_fz, lens)
        out_rx = self.model_rx(x_rx, lens)
        out_rz = self.model_rz(x_rz, lens)
        zero_padding_loc = (out_fx == 0.) & (out_fz == 0.) & (out_rx == 0.) & (out_rz == 0.)
        out_fx = self.inverse_scaling(out_fx, 'midout_force_x')
        out_fz = self.inverse_scaling(out_fz, 'midout_force_z')
        out_rx = self.inverse_scaling(out_rx, 'midout_r_x')
        out_rz = self.inverse_scaling(out_rz, 'midout_r_z')
        weight = anthro[:, 0, 0].unsqueeze(1).unsqueeze(2)
        height = anthro[:, 0, 1].unsqueeze(1).unsqueeze(2)
        output = out_fx * out_rz - out_fz * out_rx
        output = torch.div(output, weight * height)
        output[zero_padding_loc] = 0
        return output

    def inverse_scaling(self, data, fields):
        data[data == 0.] = np.nan
        if isinstance(self.scalars[fields], MinMaxScaler):
            bias_, scale_ = self.scalars[fields].min_[0], self.scalars[fields].scale_[0]
        elif isinstance(self.scalars[fields], StandardScaler):
            bias_, scale_ = self.scalars[fields].mean_[0], self.scalars[fields].scale_[0]
        data = torch.add(data, - bias_)
        data = torch.div(data, scale_)
        data[torch.isnan(data)] = 0.
        return data


# define preprocess functions
def make_joints_relative_to_midhip():
    midhip_col_loc = [data_fields.index('MidHip' + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
    key_points_to_process = ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
    for sub_name, sub_data in data_all_sub.items():
        midhip_90_and_180_data = sub_data[:, :, midhip_col_loc]
        for key_point in key_points_to_process:
            key_point_col_loc = [data_fields.index(key_point + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
            sub_data[:, :, key_point_col_loc] = sub_data[:, :, key_point_col_loc] - midhip_90_and_180_data
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
    # Five models are available: 8IMU_camera, 3IMU_camera, 8IMU, 3IMU, camera
    model_name = '8IMU_camera'
    # Two target moments: KAM or KFM
    target_moment = 'KAM'

    assert model_name in ['8IMU_camera', '3IMU_camera', '8IMU', '3IMU', 'camera'], 'Incorrect model name.'
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

    # subject_01 or subject_02 are available;
    # subject_01's data was involved in model training, while subject_02's data was not
    subject_data = data_all_sub['subject_01']
    model_inputs = {}
    model_inputs['anthro'] = torch.from_numpy(subject_data[:, :, [data_fields.index('body weight'),
                                                                  data_fields.index('body height')]])
    model_inputs['step_length'] = torch.from_numpy(np.sum(~(subject_data[:, :, 0] == 0.), axis=1))

    for submodel, component in zip([model.model_fx, model.model_fz, model.model_rx, model.model_rz],
                                   ['force_x', 'force_z', 'r_x', 'r_z']):
        input_fields_ = submodel.input_fields
        data_to_process = copy.deepcopy(subject_data)

        other_feature_loc = [data_fields.index(field) for field in input_fields_ if 'Acc' not in field]
        data_to_process[:, :, other_feature_loc] = normalize_array_separately(
            data_to_process[:, :, other_feature_loc], model.scalars[component + '_other'], 'transform', scalar_mode='by_each_column')

        weighted_acc_loc = [data_fields.index(field) for field in input_fields_ if 'Acc' in field]
        if len(weighted_acc_loc) > 0:
            data_to_process[:, :, weighted_acc_loc] = normalize_array_separately(
                data_to_process[:, :, weighted_acc_loc], model.scalars[component + '_acc'], 'transform', scalar_mode='by_all_columns')
        submodel_input = data_to_process[:, :, [data_fields.index(field) for field in input_fields_]]
        model_inputs[component] = torch.from_numpy(submodel_input)

    """ step 2: predict moment of subject 01 """
    predicted = model(model_inputs['force_x'], model_inputs['force_z'], model_inputs['r_x'], model_inputs['r_z'],
                      model_inputs['anthro'], model_inputs['step_length']).detach().numpy()
    if target_moment == 'KFM':
        predicted = - predicted

    """ step 3: plot estimation and true values """
    if target_moment == 'KAM':
        ground_truth_moment = subject_data[:, :, data_fields.index('EXT_KM_Y')]
    else:
        ground_truth_moment = -subject_data[:, :, data_fields.index('EXT_KM_X')]
    # change unit of body weight from Kg to N
    ground_truth_moment, predicted = GRAVITY * ground_truth_moment, GRAVITY * predicted
    plt.figure()
    plt.plot(ground_truth_moment.ravel(), label='True Value')
    plt.plot(predicted.ravel(), label='Predicted Value')
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel('Time Step')
    ax.set_ylabel(target_moment + ' (BW X BH)')
    plt.title(model_name + ' model')
    plt.show()
