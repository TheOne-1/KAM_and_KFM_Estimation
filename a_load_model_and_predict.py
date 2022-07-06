import h5py
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import json
import matplotlib.pyplot as plt
import numpy as np

# define consts
KEYPOINTS = ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
JOINTS_90_FIELDS = [loc + axis + '_90' for loc in KEYPOINTS for axis in ['_x', '_y']]
JOINTS_180_FIELDS = [loc + axis + '_180' for loc in KEYPOINTS for axis in ['_x', '_y']]
VID_ALL = JOINTS_90_FIELDS + JOINTS_180_FIELDS
IMU_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
ACC_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[:3]]
GYR_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[3:]]
ACC_3IMU = [field + '_' + sensor for sensor in ['L_FOOT', 'R_FOOT', 'WAIST'] for field in IMU_FIELDS[:3]]
GYR_3IMU = [field + '_' + sensor for sensor in ['L_FOOT', 'R_FOOT', 'WAIST'] for field in IMU_FIELDS[3:]]

LSTM_UNITS, FCNN_UNITS = 40, 40
WEIGHT_LOC, HEIGHT_LOC = range(2)
GRAVITY = 9.81


class InertialNet(torch.nn.Module):
    def __init__(self, x_dim, net_name, seed=0, nlayer=1):
        super(InertialNet, self).__init__()
        self.net_name = net_name
        torch.manual_seed(seed)
        self.rnn_layer = torch.nn.LSTM(x_dim, LSTM_UNITS, nlayer, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(0.2)
        for name, param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        sequence, _ = self.rnn_layer(sequence)
        sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True, total_length=152)
        sequence = self.dropout(sequence)
        return sequence


class OutNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(OutNet, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, FCNN_UNITS, bias=True)
        self.linear_2 = torch.nn.Linear(FCNN_UNITS, 2, bias=True)
        self.relu = torch.nn.ReLU()
        for layer in [self.linear_1, self.linear_2]:
            torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, others):
        sequence = self.linear_1(sequence)
        sequence = self.relu(sequence)
        sequence = self.linear_2(sequence)
        weight = others[:, 0, WEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        height = others[:, 0, HEIGHT_LOC].unsqueeze(1).unsqueeze(2)
        sequence = torch.div(sequence, weight * GRAVITY * height / 100)
        return sequence


class VideoNet(InertialNet):
    pass


class LmfNet(nn.Module):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self, acc_dim, gyr_dim):
        super(LmfNet, self).__init__()
        self.acc_subnet = InertialNet(acc_dim, 'acc net', seed=0)
        self.gyr_subnet = InertialNet(gyr_dim, 'gyr net', seed=0)
        self.vid_subnet = VideoNet(24, 'vid net', seed=0)
        self.rank = 10
        self.fused_dim = 40

        self.acc_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
        self.gyr_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
        self.vid_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim))

        # init factors
        nn.init.xavier_normal_(self.acc_factor, 10)
        nn.init.xavier_normal_(self.gyr_factor, 10)
        nn.init.xavier_normal_(self.vid_factor, 10)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        self.out_net = OutNet(self.fused_dim)

    def __str__(self):
        return 'LMF fusion net'

    def set_scalars(self, scalars):
        self.scalars = scalars

    def forward(self, acc_x, gyr_x, vid_x, others, lens):
        acc_h = self.acc_subnet(acc_x, lens)
        gyr_h = self.gyr_subnet(gyr_x, lens)
        vid_h = self.vid_subnet(vid_x, lens)
        batch_size = acc_h.data.shape[0]
        data_type = torch.FloatTensor

        _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).type(data_type), requires_grad=False), acc_h), dim=2)
        _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).type(data_type), requires_grad=False), gyr_h), dim=2)
        _vid_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, vid_x.shape[1], 1).type(data_type), requires_grad=False), vid_h), dim=2)

        fusion_acc = torch.matmul(_acc_h, self.acc_factor)
        fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
        fusion_vid = torch.matmul(_vid_h, self.vid_factor)
        fusion_zy = fusion_acc * fusion_gyr * fusion_vid

        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias
        sequence = self.out_net(sequence, others)
        return sequence


class LmfImuOnlyNet(LmfNet):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self, acc_dim, gyr_dim):
        super(LmfImuOnlyNet, self).__init__(acc_dim, gyr_dim)
        if acc_dim <= 3:
            self.out_net = OutNet(self.fused_dim)  # do not use high level features
        else:
            self.out_net = OutNet(self.fused_dim)  # only use FPA from high level features

    def __str__(self):
        return 'LMF IMU only net'

    def forward(self, acc_x, gyr_x, vid_x, others, lens):
        acc_h = self.acc_subnet(acc_x, lens)
        gyr_h = self.gyr_subnet(gyr_x, lens)
        batch_size = acc_h.data.shape[0]
        data_type = torch.FloatTensor

        _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).type(data_type), requires_grad=False), acc_h), dim=2)
        _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).type(data_type), requires_grad=False), gyr_h), dim=2)

        fusion_acc = torch.matmul(_acc_h, self.acc_factor)
        fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
        fusion_vid = torch.full_like(fusion_acc, 1)
        fusion_zy = fusion_acc * fusion_gyr * fusion_vid
        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias
        sequence = self.out_net(sequence, others)
        return sequence


class LmfCameraOnlyNet(LmfNet):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self):
        super(LmfCameraOnlyNet, self).__init__(1, 1)
        self.out_net = OutNet(self.fused_dim, [4])

    def __str__(self):
        return 'LMF camera only net'

    def forward(self, acc_x, gyr_x, vid_x, others, lens):
        vid_h = self.vid_subnet(vid_x, lens)
        batch_size = vid_h.data.shape[0]
        data_type = torch.FloatTensor
        _vid_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, vid_x.shape[1], 1).type(data_type), requires_grad=False), vid_h), dim=2)
        fusion_vid = torch.matmul(_vid_h, self.vid_factor)
        fusion_zy = fusion_vid
        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias
        sequence = self.out_net(sequence, others)
        return sequence


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


def normalize_vid_by_size_of_subject_in_static_trial():
    for sub_name, sub_data in data_all_sub.items():
        height_col_loc = data_fields.index('body height')
        sub_height = sub_data[0, 0, height_col_loc]
        for camera in ['90', '180']:
            vid_col_loc = [data_fields.index(keypoint + axis + camera) for keypoint in KEYPOINTS for axis in ['_x_', '_y_']]
            sub_data[:, :, vid_col_loc] = sub_data[:, :, vid_col_loc] / sub_height
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


if __name__ == "__main__":
    """ step 0: select model and load data """
    # Five models are available: 8IMU_camera, 3IMU_camera, 8IMU, 3IMU, camera
    model_name = '8IMU_camera'
    assert model_name in ['8IMU_camera', '3IMU_camera', '8IMU', '3IMU', 'camera'], 'Incorrect model name.'

    # one example data file is available
    with h5py.File('trained_models_and_example_data/example_data.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    if '8' in model_name:
        acc_col_loc = [data_fields.index(field) for field in ACC_ALL]
        gyr_col_loc = [data_fields.index(field) for field in GYR_ALL]
    elif '3' in model_name:
        acc_col_loc = [data_fields.index(field) for field in ACC_3IMU]
        gyr_col_loc = [data_fields.index(field) for field in GYR_3IMU]
    else:
        acc_col_loc = [0]
        gyr_col_loc = [0]

    if 'camera' in model_name:
        vid_col_loc = [data_fields.index(field) for field in VID_ALL]
    else:
        vid_col_loc = [0]

    model_path = './trained_models_and_example_data/' + model_name + '.pth'
    model = torch.load(model_path)

    """ step 1: prepare subject 01's data as input """
    make_joints_relative_to_midhip()

    # subject_01 or subject_02 are available;
    # subject_01's data was not involved in model training, while subject_02's data was involved
    subject_data = data_all_sub['subject_02']
    model_inputs = {}
    model_inputs['others'] = torch.from_numpy(subject_data[:, :, [data_fields.index('body weight'),
                                                                  data_fields.index('body height')]])
    model_inputs['step_length'] = torch.from_numpy(np.sum(~(subject_data[:, :, 0] == 0.), axis=1))

    subject_data[:, :, acc_col_loc] = normalize_array_separately(
        subject_data[:, :, acc_col_loc], model.scalars['input_acc'], 'transform')
    model_inputs['input_acc'] = torch.from_numpy(subject_data[:, :, acc_col_loc])
    subject_data[:, :, gyr_col_loc] = normalize_array_separately(
        subject_data[:, :, gyr_col_loc], model.scalars['input_gyr'], 'transform')
    model_inputs['input_gyr'] = torch.from_numpy(subject_data[:, :, gyr_col_loc])
    subject_data[:, :, vid_col_loc] = normalize_array_separately(
        subject_data[:, :, vid_col_loc], model.scalars['input_vid'], 'transform')
    model_inputs['input_vid'] = torch.from_numpy(subject_data[:, :, vid_col_loc])

    """ step 2: predict moment via a torch model """
    pred = model(model_inputs['input_acc'], model_inputs['input_gyr'], model_inputs['input_vid'],
                 model_inputs['others'], model_inputs['step_length']).detach().numpy()

    """ step 3: plot estimation and true values """
    ground_truth_moment = subject_data[:, :, [data_fields.index(field) for field in ('EXT_KM_X', 'EXT_KM_Y')]]
    plt.figure()
    colors = ['C1', 'C2', 'C3']
    plt.plot(ground_truth_moment[:, :, 0].ravel(), '--', c='C0', label='True KFM')
    plt.plot(pred[:, :, 0].ravel(), c='C0', label='Predicted KFM')
    plt.plot(ground_truth_moment[:, :, 1].ravel(), '--', c='C1', label='True KAM')
    plt.plot(pred[:, :, 1].ravel(), c='C1', label='Predicted KAM')
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Knee Moment (BW X BH)')
    plt.title(model_name + ' model')
    plt.show()
