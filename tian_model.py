import os
import random
from base_kam_model import BaseModel
from torch.optim.lr_scheduler import ExponentialLR
import torch
import torch.nn as nn
from customized_logger import logger as logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
import h5py
from const import IMU_FIELDS, SENSOR_LIST, DATA_PATH, VIDEO_LIST, SUBJECT_WEIGHT, SUBJECT_HEIGHT, FORCE_PHASE, \
    FORCE_DATA_FIELDS, STATIC_DATA, SEGMENT_MASS_PERCENT, SUBJECT_ID, TRIAL_ID
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from types import SimpleNamespace
import pandas as pd

RKNEE_MARKER_FIELDS = [marker + axis for marker in ['RFME', 'RFLE'] for axis in ['_X', '_Y', '_Z']]
RANKLE_MARKER_FIELDS = [marker + axis for marker in ['RTAM', 'RFAL'] for axis in ['_X', '_Y', '_Z']]
RHIP_MARKER_FIELDS = [marker + axis for marker in ['RFT'] for axis in ['_X', '_Y', '_Z']]
LKNEE_MARKER_FIELDS = [marker + axis for marker in ['LFLE', 'LFME'] for axis in ['_X', '_Y', '_Z']]
LANKLE_MARKER_FIELDS = [marker + axis for marker in ['LFAL', 'LTAM'] for axis in ['_X', '_Y', '_Z']]
LHIP_MARKER_FIELDS = [marker + axis for marker in ['LFT'] for axis in ['_X', '_Y', '_Z']]
ASIS_MARKER_FIELDS = [marker + axis for marker in ['LIAS', 'RIAS'] for axis in ['_X', '_Y', '_Z']]
PSIS_MARKER_FIELDS = [marker + axis for marker in ['LIPS', 'RIPS'] for axis in ['_X', '_Y', '_Z']]
SHOULDER_MARKER_FIELDS = [marker + axis for marker in ['LAC', 'RAC'] for axis in ['_X', '_Y', '_Z']]
SPINE_MARKER_FIELDS = [marker + axis for marker in ['CV7', 'MAI'] for axis in ['_X', '_Y', '_Z']]


class TianCNN(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        kernel_num = 32
        self.conv1 = nn.Conv1d(x_dim, 8 * kernel_num, kernel_size=3, stride=1, bias=False)
        self.drop1 = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8 * kernel_num, 2 * kernel_num, kernel_size=3, stride=1, bias=False)
        self.drop2 = nn.Dropout(p=0.2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(2 * kernel_num, kernel_num, kernel_size=3, stride=1, bias=False)
        self.drop3 = nn.Dropout(p=0.2)
        self.pool3 = nn.AvgPool1d(2)
        self.conv4 = nn.Conv1d(kernel_num, kernel_num, kernel_size=3, stride=1, bias=False)
        self.drop4 = nn.Dropout(p=0.2)
        self.pool4 = nn.AvgPool1d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.conv2output = nn.Linear(64, y_dim * 152, bias=False)
        self.drop = nn.Dropout(p=0.1)
        self.y_dim = y_dim
        self.x_dim = x_dim

    def forward(self, sequence, lens):
        sequence = sequence[:, 30:100, :]  # take part of the data
        sequence.transpose_(1, 2)
        sequence = self.relu(self.conv1(sequence))
        sequence = self.drop1(sequence)
        sequence = self.pool1(sequence)
        sequence = self.relu(self.conv2(sequence))
        sequence = self.drop2(sequence)
        sequence = self.pool2(sequence)
        sequence = self.relu(self.conv3(sequence))
        sequence = self.drop3(sequence)
        sequence = self.pool3(sequence)
        sequence = self.relu(self.conv4(sequence))
        sequence = self.drop4(sequence)
        sequence = self.pool4(sequence)
        sequence = self.flatten(sequence)
        output = self.conv2output(sequence)
        output = torch.reshape(output, (-1, 152, self.y_dim))
        return output


class TianRNN(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=10, nlayer=2):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_layer = nn.LSTM(x_dim, hidden_dim, nlayer, batch_first=True, bidirectional=True)
        self.y_dim = y_dim
        self.r2d = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.d2o = nn.Linear(hidden_dim, y_dim, bias=False)
        self.relu = nn.ReLU()
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
        output = out_fx * out_rz - out_fz * out_rx
        weight = anthro[:, 0, 0].unsqueeze(1).unsqueeze(2)
        height = anthro[:, 0, 1].unsqueeze(1).unsqueeze(2)
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


class TianModel(BaseModel):
    def __init__(self,  *args, **kwargs):
        BaseModel.__init__(self,  *args, **kwargs)
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3
        self.vid_static_cali()
        self.get_relative_vid_vector()
        self.get_rotated_gravityfree_body_weighted_imu()
        self.add_additional_columns()
        self.iter = 0

    def vid_static_cali(self):
        vid_y_90_col_loc = [self._data_fields.index(marker + '_y_90') for marker in VIDEO_LIST]
        # knee_op_col_loc = self._data_fields.index('RKnee_y_90')
        # knee_vi_col_loc = self._data_fields.index('RFLE_Z')
        for sub_name, sub_data in self._data_all_sub.items():
            # plt.figure()
            # plt.plot(sub_data[:, :, knee_op_col_loc].ravel(), sub_data[:, :, knee_vi_col_loc].ravel(), '.')
            # plt.plot(sub_data[:, :, knee_op_col_loc].ravel())
            # plt.plot(sub_data[:, :, knee_vi_col_loc].ravel())
            static_side_df = pd.read_csv(DATA_PATH + '/' + sub_name + '/combined/static_side.csv', index_col=0)
            r_ankle_z = np.median(static_side_df['RAnkle_y_90'])
            sub_data[:, :, vid_y_90_col_loc] = sub_data[:, :, vid_y_90_col_loc] - r_ankle_z
            self._data_all_sub[sub_name] = sub_data
        # plt.show()

    def get_relative_vid_vector(self, scale_180=False):
        midhip_col_loc = [self._data_fields.index('MidHip' + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
        key_points_to_process = ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
        for sub_name, sub_data in self._data_all_sub.items():
            midhip_180_data = sub_data[:, :, midhip_col_loc]
            for key_point in key_points_to_process:
                key_point_col_loc = [self._data_fields.index(key_point + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
                sub_data[:, :, key_point_col_loc] = sub_data[:, :, key_point_col_loc] - midhip_180_data
            self._data_all_sub[sub_name] = sub_data
        if scale_180:
            self.scale_vid_180_vectors_via_vid_90()

    def scale_vid_180_vectors_via_vid_90(self):
        midhip_90_x_loc = self._data_fields.index('MidHip_x_90')
        key_points_to_process = ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
        for sub_name, sub_data in self._data_all_sub.items():
            midhip_90_x = sub_data[:, :, midhip_90_x_loc]
            for key_point in key_points_to_process:
                key_point_col_loc = [self._data_fields.index(key_point + axis + angle) for axis in ['_x', '_y'] for angle in ['_180']]
                sub_data[:, :, key_point_col_loc[0]] = sub_data[:, :, key_point_col_loc[0]] / (2060 - midhip_90_x)
                sub_data[:, :, key_point_col_loc[1]] = sub_data[:, :, key_point_col_loc[1]] / (2060 - midhip_90_x)
            self._data_all_sub[sub_name] = sub_data

    def add_additional_columns(self):
        marker_rknee_col_loc = [self._data_fields.index(field_name) for field_name in RKNEE_MARKER_FIELDS]
        force_col_loc = [self._data_fields.index(field_name) for field_name in FORCE_DATA_FIELDS]
        for sub_name, sub_data in self._data_all_sub.items():
            marker_data = sub_data[:, :, marker_rknee_col_loc].copy()
            force_data = sub_data[:, :, force_col_loc].copy()
            knee_vector = force_data[:, :, 9:12] - (marker_data[:, :, :3] + marker_data[:, :, 3:6]) / 2
            self._data_all_sub[sub_name] = np.concatenate([sub_data, knee_vector], axis=2)
        self._data_fields.extend(['KNEE_X', 'KNEE_Y', 'KNEE_Z'])

    def get_rotated_gravityfree_body_weighted_imu(self, ROTATE_IMU=False):
        def transform_segment_imu(segment_imu, segment_ml, segment_y, segment_z=None):
            data_shape_ori = segment_imu.shape
            segment_acc, segment_gyr = segment_imu[:, :, :3].reshape([-1, 3]), segment_imu[:, :, 3:].reshape([-1, 3])
            segment_ml = segment_ml.reshape([-1, 3])
            if segment_z is None:
                segment_y = segment_y.reshape([-1, 3])
                segment_z = np.cross(segment_ml, segment_y)
                segment_x = np.cross(segment_y, segment_z)
            else:
                segment_z = segment_z.reshape([-1, 3])
                segment_y = np.cross(segment_z, segment_ml)
                segment_x = np.cross(segment_y, segment_z)

            fun_norm_vect = lambda v: v / np.linalg.norm(v)
            segment_x = np.apply_along_axis(fun_norm_vect, 1, segment_x)
            segment_y = np.apply_along_axis(fun_norm_vect, 1, segment_y)
            segment_z = np.apply_along_axis(fun_norm_vect, 1, segment_z)

            dcm_mat = np.array([segment_x, segment_y, segment_z])
            dcm_mat = np.swapaxes(dcm_mat, 0, 1)
            segment_acc_rotated = np.array(list(map(np.matmul, dcm_mat, segment_acc)))
            segment_gyr_rotated = np.array(list(map(np.matmul, dcm_mat, segment_gyr)))
            segment_imu_transformed = np.column_stack([segment_acc_rotated, segment_gyr_rotated]).reshape(data_shape_ori)
            segment_imu_transformed[np.isnan(segment_imu_transformed)] = 0
            return segment_imu_transformed
        imu_lfoot_col_loc = [self._data_fields.index(field + '_L_FOOT') for field in IMU_FIELDS[:6]]
        imu_lshank_col_loc = [self._data_fields.index(field + '_L_SHANK') for field in IMU_FIELDS[:6]]
        imu_lthigh_col_loc = [self._data_fields.index(field + '_L_SHANK') for field in IMU_FIELDS[:6]]
        imu_rfoot_col_loc = [self._data_fields.index(field + '_R_FOOT') for field in IMU_FIELDS[:6]]
        imu_rshank_col_loc = [self._data_fields.index(field + '_R_SHANK') for field in IMU_FIELDS[:6]]
        imu_rthigh_col_loc = [self._data_fields.index(field + '_R_SHANK') for field in IMU_FIELDS[:6]]
        imu_pelvis_col_loc = [self._data_fields.index(field + '_WAIST') for field in IMU_FIELDS[:6]]
        imu_trunk_col_loc = [self._data_fields.index(field + '_CHEST') for field in IMU_FIELDS[:6]]

        marker_lknee_col_loc = [self._data_fields.index(field_name) for field_name in LKNEE_MARKER_FIELDS]
        marker_lankle_col_loc = [self._data_fields.index(field_name) for field_name in LANKLE_MARKER_FIELDS]
        marker_lhip_col_loc = [self._data_fields.index(field_name) for field_name in LHIP_MARKER_FIELDS]
        marker_rknee_col_loc = [self._data_fields.index(field_name) for field_name in RKNEE_MARKER_FIELDS]
        marker_rankle_col_loc = [self._data_fields.index(field_name) for field_name in RANKLE_MARKER_FIELDS]
        marker_rhip_col_loc = [self._data_fields.index(field_name) for field_name in RHIP_MARKER_FIELDS]
        marker_asis_col_loc = [self._data_fields.index(field_name) for field_name in ASIS_MARKER_FIELDS]
        marker_psis_col_loc = [self._data_fields.index(field_name) for field_name in PSIS_MARKER_FIELDS]
        marker_shoulder_col_loc = [self._data_fields.index(field_name) for field_name in SHOULDER_MARKER_FIELDS]
        marker_spine_col_loc = [self._data_fields.index(field_name) for field_name in SPINE_MARKER_FIELDS]

        weight_col_loc = self._data_fields.index(SUBJECT_WEIGHT)
        for sub_name, sub_data in self._data_all_sub.items():
            if ROTATE_IMU:
                lshank_ml = sub_data[:, :, marker_lknee_col_loc[:3]] - sub_data[:, :, marker_lknee_col_loc[3:]]
                lshank_y = (sub_data[:, :, marker_lknee_col_loc[:3]] + sub_data[:, :, marker_lknee_col_loc[3:]]) / 2 - \
                           (sub_data[:, :, marker_lankle_col_loc[:3]] + sub_data[:, :, marker_lankle_col_loc[3:]]) / 2
                sub_data[:, :, imu_lshank_col_loc] = transform_segment_imu(sub_data[:, :, imu_lshank_col_loc], lshank_ml, lshank_y)
                lthigh_ml = lshank_ml
                lthigh_y = sub_data[:, :, marker_lhip_col_loc] - \
                           (sub_data[:, :, marker_lknee_col_loc[:3]] + sub_data[:, :, marker_lknee_col_loc[3:]]) / 2
                sub_data[:, :, imu_lthigh_col_loc] = transform_segment_imu(sub_data[:, :, imu_lthigh_col_loc], lthigh_ml, lthigh_y)
                trunk_ml = sub_data[:, :, marker_shoulder_col_loc[:3]] - sub_data[:, :, marker_shoulder_col_loc[3:]]
                trunk_y = sub_data[:, :, marker_spine_col_loc[:3]] - sub_data[:, :, marker_spine_col_loc[3:]]
                sub_data[:, :, imu_trunk_col_loc] = transform_segment_imu(sub_data[:, :, imu_trunk_col_loc], trunk_ml, trunk_y)
                pelvis_ml = sub_data[:, :, marker_asis_col_loc[:3]] - sub_data[:, :, marker_asis_col_loc[3:]]
                pelvis_z = (sub_data[:, :, marker_asis_col_loc[:3]] + sub_data[:, :, marker_asis_col_loc[3:]]) / 2 - \
                           (sub_data[:, :, marker_psis_col_loc[:3]] + sub_data[:, :, marker_psis_col_loc[3:]]) / 2
                sub_data[:, :, imu_pelvis_col_loc] = transform_segment_imu(sub_data[:, :, imu_pelvis_col_loc], pelvis_ml, None, pelvis_z)
                rshank_ml = sub_data[:, :, marker_rknee_col_loc[:3]] - sub_data[:, :, marker_rknee_col_loc[3:]]
                rshank_y = (sub_data[:, :, marker_rknee_col_loc[:3]] + sub_data[:, :, marker_rknee_col_loc[3:]]) / 2 - \
                          (sub_data[:, :, marker_rankle_col_loc[:3]] + sub_data[:, :, marker_rankle_col_loc[3:]]) / 2
                sub_data[:, :, imu_rshank_col_loc] = transform_segment_imu(sub_data[:, :, imu_rshank_col_loc], rshank_ml, rshank_y)
                rthigh_ml = rshank_ml
                rthigh_y = sub_data[:, :, marker_rhip_col_loc] - \
                          (sub_data[:, :, marker_rknee_col_loc[:3]] + sub_data[:, :, marker_rknee_col_loc[3:]]) / 2
                sub_data[:, :, imu_rthigh_col_loc] = transform_segment_imu(sub_data[:, :, imu_rthigh_col_loc], rthigh_ml, rthigh_y)

            sub_weight = sub_data[0, 0, weight_col_loc]
            sub_data[:, :, imu_lfoot_col_loc[:3]] = sub_data[:, :, imu_lfoot_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['L_FOOT'] / 100
            sub_data[:, :, imu_lshank_col_loc[:3]] = sub_data[:, :, imu_lshank_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['L_SHANK'] / 100
            sub_data[:, :, imu_lthigh_col_loc[:3]] = sub_data[:, :, imu_lthigh_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['L_THIGH'] / 100
            sub_data[:, :, imu_rfoot_col_loc[:3]] = sub_data[:, :, imu_rfoot_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['R_FOOT'] / 100
            sub_data[:, :, imu_rshank_col_loc[:3]] = sub_data[:, :, imu_rshank_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['R_SHANK'] / 100
            sub_data[:, :, imu_rthigh_col_loc[:3]] = sub_data[:, :, imu_rthigh_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['R_THIGH'] / 100
            sub_data[:, :, imu_pelvis_col_loc[:3]] = sub_data[:, :, imu_pelvis_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['WAIST'] / 100
            sub_data[:, :, imu_trunk_col_loc[:3]] = sub_data[:, :, imu_trunk_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT['CHEST'] / 100

            self._data_all_sub[sub_name] = sub_data

    def preprocess_train_data(self, x, y, weight):
        y['midout_force_x'], y['midout_force_z'] = -y['midout_force_x'], -y['midout_force_z']
        y['midout_r_x'], y['midout_r_z'] = y['midout_r_x'] / 1000, y['midout_r_z'] / 1000
        x_need_norm = {k: x[k] for k in set(list(x.keys())) - set(['anthro'])}
        x.update(
            **self.normalize_data(x_need_norm, self._data_scalar, 'fit_transform', scalar_mode='by_each_column'))
        y_need_norm = {k: y[k] for k in set(list(y.keys())) - set(['main_output', 'auxiliary_info'])}
        y.update(
            **self.normalize_data(y_need_norm, self._data_scalar, 'fit_transform', scalar_mode='by_each_column'))
        return x, y, weight

    def preprocess_validation_test_data(self, x, y, weight):
        y['midout_force_x'], y['midout_force_z'] = -y['midout_force_x'], -y['midout_force_z']
        y['midout_r_x'], y['midout_r_z'] = y['midout_r_x'] / 1000, y['midout_r_z'] / 1000
        x_need_norm = {k: x[k] for k in set(list(x.keys())) - set(['anthro'])}
        x.update(**self.normalize_data(x_need_norm, self._data_scalar, 'transform', scalar_mode='by_each_column'))
        y_need_norm = {k: y[k] for k in set(list(y.keys())) - set(['main_output', 'auxiliary_info'])}
        y.update(**self.normalize_data(y_need_norm, self._data_scalar, 'transform', scalar_mode='by_each_column'))
        return x, y, weight

    @staticmethod
    def loss_fun_valid_part(y_pred, y, left, right):
        y_pred_valid = y_pred[:, left:-right]
        y_valid = y[:, left:-right]
        loss_positive = (y_pred_valid - y_valid).pow(2).sum()
        return loss_positive

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        sub_model_base_param = {'epoch': 5, 'batch_size': 20, 'lr': 3e-3, 'weight_decay': 3e-4, 'use_ratio': 100}
        self.train_step_lens, self.validation_step_lens = self._get_step_len(x_train), self._get_step_len(x_validation)

        x_train_rx, x_validation_rx = x_train['r_x'], x_validation['r_x']
        y_train_rx, y_validation_rx = y_train['midout_r_x'], y_validation['midout_r_x']
        model_rx = TianRNN(x_train_rx.shape[2], y_train_rx.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_r_x', 'fields': ['KNEE_X']}}
        self.build_sub_model(model_rx, x_train_rx, y_train_rx, x_validation_rx, y_validation_rx, validation_weight, params)

        x_train_rz, x_validation_rz = x_train['r_z'], x_validation['r_z']
        y_train_rz, y_validation_rz = y_train['midout_r_z'], y_validation['midout_r_z']
        model_rz = TianRNN(x_train_rz.shape[2], y_train_rz.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_r_z', 'fields': ['KNEE_Z']}}
        self.build_sub_model(model_rz, x_train_rz, y_train_rz, x_validation_rz, y_validation_rz, validation_weight, params)

        x_train_fz, x_validation_fz = x_train['force_z'], x_validation['force_z']
        y_train_fz, y_validation_fz = y_train['midout_force_z'], y_validation['midout_force_z']
        model_fz = TianRNN(x_train_fz.shape[2], y_train_fz.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_force_z', 'fields': ['plate_2_force_z']}}
        self.build_sub_model(model_fz, x_train_fz, y_train_fz, x_validation_fz, y_validation_fz, validation_weight, params)

        x_train_fx, x_validation_fx = x_train['force_x'], x_validation['force_x']
        y_train_fx, y_validation_fx = y_train['midout_force_x'], y_validation['midout_force_x']
        model_fx = TianRNN(x_train_fx.shape[2], y_train_fx.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_force_x', 'fields': ['plate_2_force_x']}}
        self.build_sub_model(model_fx, x_train_fx, y_train_fx, x_validation_fx, y_validation_fx, validation_weight, params)

        model_fx_pre = type(model_fx)(x_train_fx.shape[2], y_train_fx.shape[2]).cuda()
        model_fz_pre = type(model_fz)(x_train_fz.shape[2], y_train_fz.shape[2]).cuda()
        model_rx_pre = type(model_rx)(x_train_rx.shape[2], y_train_rx.shape[2]).cuda()
        model_rz_pre = type(model_rz)(x_train_rz.shape[2], y_train_rz.shape[2]).cuda()
        model_fx_pre.load_state_dict(model_fx.state_dict())
        model_fz_pre.load_state_dict(model_fz.state_dict())
        model_rx_pre.load_state_dict(model_rx.state_dict())
        model_rz_pre.load_state_dict(model_rz.state_dict())

        four_source_model = FourSourceModel(model_fx, model_fz, model_rx, model_rz, self._data_scalar)
        params = {**sub_model_base_param, **{'target_name': 'main_output', 'fields': ['EXT_KM_Y']}}
        params['lr'] = params['lr'] * 0.1
        params['batch_size'] = 200
        self.build_main_model(four_source_model, x_train, y_train, x_validation, y_validation, validation_weight,
                              params)
        res_models = {'four_source_model': four_source_model, 'model_rx_pre': model_rx_pre,
                      'model_rz_pre': model_rz_pre, 'model_fx_pre': model_fx_pre, 'model_fz_pre': model_fz_pre,
                      'model_rx': model_rx, 'model_rz': model_rz, 'model_fx': model_fx, 'model_fz': model_fz}
        return res_models

    def build_main_model(self, model, x_train, y_train, x_validation, y_validation, validation_weight, params):
        def prepare_main_model_data(x_train, y_train, train_step_lens, x_validation, y_validation, validation_step_lens,
                                    batch_size):
            x_fx, x_fz, x_rx, x_rz, x_anthro = x_train['force_x'], x_train['force_z'], x_train['r_x'], x_train['r_z'], \
                                               x_train['anthro']
            y_train = torch.from_numpy(y_train['main_output']).float().cuda()
            x_fx, x_fz = torch.from_numpy(x_fx).float().cuda(), torch.from_numpy(x_fz).float().cuda()
            x_rx, x_rz, x_anthro = torch.from_numpy(x_rx).float().cuda(), torch.from_numpy(
                x_rz).float().cuda(), torch.from_numpy(x_anthro).float().cuda()
            train_step_lens = torch.from_numpy(train_step_lens)
            train_ds = TensorDataset(x_fx, x_fz, x_rx, x_rz, x_anthro, y_train, train_step_lens)
            train_size = int(0.9 * len(train_ds))
            vali_from_train_size = len(train_ds) - train_size
            train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds,
                                                                                 [train_size, vali_from_train_size])
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=batch_size)

            x_fx, x_fz, x_rx, x_rz, x_anthro = x_validation['force_x'], x_validation['force_z'], x_validation['r_x'], \
                                               x_validation['r_z'], x_validation['anthro']
            y_validation = torch.from_numpy(y_validation['main_output']).float().cuda()
            x_fx, x_fz = torch.from_numpy(x_fx).float().cuda(), torch.from_numpy(x_fz).float().cuda()
            x_rx, x_rz, x_anthro = torch.from_numpy(x_rx).float().cuda(), torch.from_numpy(
                x_rz).float().cuda(), torch.from_numpy(x_anthro).float().cuda()
            vali_step_lens = torch.from_numpy(validation_step_lens)
            test_ds = TensorDataset(x_fx, x_fz, x_rx, x_rz, x_anthro, y_validation, vali_step_lens)
            test_dl = DataLoader(test_ds, batch_size=batch_size)
            vali_from_test_ds = TensorDataset(x_fx, x_fz, x_rx, x_rz, x_anthro, y_validation, vali_step_lens)
            num_of_step_for_peek = int(0.1 * len(y_validation))
            vali_from_test_ds, _ = torch.utils.data.dataset.random_split(vali_from_test_ds, [num_of_step_for_peek, len(
                y_validation) - num_of_step_for_peek])
            vali_from_test_dl = DataLoader(vali_from_test_ds, batch_size=batch_size)
            return train_dl, vali_from_train_dl, vali_from_test_dl, test_dl

        def train(model, train_dl, optimizer, loss_fn, params):
            for i_batch, (xb_0, xb_1, xb_2, xb_3, xb_4, yb, lens) in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > params.use_ratio:
                    continue  # increase the speed of epoch
                optimizer.zero_grad()
                y_pred = model(xb_0, xb_1, xb_2, xb_3, xb_4, lens)
                train_loss = loss_fn(y_pred, yb)
                train_loss.backward()
                optimizer.step()

        def eval_after_training(model, test_dl, y_validation, validation_weight, params, show_plots=False):
            with torch.no_grad():
                y_pred_list = []
                for i_batch, (xb_0, xb_1, xb_2, xb_3, xb_4, yb, lens) in enumerate(test_dl):
                    y_pred_list.append(model(xb_0, xb_1, xb_2, xb_3, xb_4, lens).detach().cpu())
                y_pred = torch.cat(y_pred_list)
            y_pred = {params.target_name: y_pred.detach().cpu().numpy()}
            all_scores = BaseModel.get_all_scores(y_validation, y_pred, {params.target_name: params.fields},
                                                  validation_weight)
            all_scores = [{'subject': 'all', **scores} for scores in all_scores]
            self.print_table(all_scores)
            if show_plots:
                self.customized_analysis(y_validation, y_pred, all_scores)
                plt.show()
            return y_pred

        def eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch):
            def vali_set_loss(nn_model, validation_dl, loss_fn):
                validation_loss = []
                for xb_0, xb_1, xb_2, xb_3, xb_4, yb, lens in validation_dl:
                    with torch.no_grad():
                        yb_pred = nn_model(xb_0, xb_1, xb_2, xb_3, xb_4, lens)
                        validation_loss.append(loss_fn(yb_pred, yb).item() / xb_0.shape[0])
                return np.mean(validation_loss)

            vali_from_train_loss = vali_set_loss(model, vali_from_train_dl, loss_fn)
            vali_from_test_loss = vali_set_loss(model, vali_from_test_dl, loss_fn)
            logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t".format(
                i_epoch, vali_from_train_loss, vali_from_test_loss, time.time() - epoch_end_time))

        params = SimpleNamespace(**params)
        train_dl, vali_from_train_dl, vali_from_test_dl, test_dl = prepare_main_model_data(
            x_train, y_train, self.train_step_lens, x_validation, y_validation, self.validation_step_lens,
            params.batch_size)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=1)
        logging.info('\tEpoch | Vali_train_Loss | Vali_test_Loss | Duration\t\t')
        epoch_end_time = time.time()
        for i_epoch in range(params.epoch):
            eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch)
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, loss_fn, params)
            scheduler.step()
        eval_after_training(model, test_dl, y_validation, validation_weight, params)

    def build_sub_model(self, model, x_train, y_train, x_validation, y_validation, validation_weight, params):
        def prepare_sub_model_data(x_train, y_train, train_step_lens, x_validation, y_validation, validation_step_lens,
                                   batch_size):
            x_train = torch.from_numpy(x_train).float()
            y_train = torch.from_numpy(y_train).float()
            train_step_lens = torch.from_numpy(train_step_lens)
            train_ds = TensorDataset(x_train, y_train, train_step_lens)
            train_size = int(0.9 * len(train_ds))
            vali_from_train_size = len(train_ds) - train_size
            train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds,
                                                                                 [train_size, vali_from_train_size])
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=batch_size)

            x_validation = torch.from_numpy(x_validation).float()
            y_validation = torch.from_numpy(y_validation).float()
            vali_step_lens = torch.from_numpy(validation_step_lens)
            test_ds = TensorDataset(x_validation, y_validation, vali_step_lens)
            test_dl = DataLoader(test_ds, batch_size=batch_size)
            vali_from_test_ds = TensorDataset(x_validation, y_validation, vali_step_lens)
            num_of_step_for_peek = int(0.1 * len(x_validation))
            vali_from_test_ds, _ = torch.utils.data.dataset.random_split(vali_from_test_ds, [num_of_step_for_peek, len(
                x_validation) - num_of_step_for_peek])
            vali_from_test_dl = DataLoader(vali_from_test_ds, batch_size=batch_size)

            return train_dl, vali_from_train_dl, vali_from_test_dl, test_dl

        def train(model, train_dl, optimizer, loss_fn, params):
            for i_batch, (xb, yb, lens) in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > params.use_ratio:
                    continue  # increase the speed of epoch
                xb = xb.cuda()
                yb = yb.cuda()
                optimizer.zero_grad()
                y_pred = model(xb, lens)
                train_loss = loss_fn(y_pred, yb)
                train_loss.backward()
                optimizer.step()

        def eval_after_training(model, test_dl, y_validation, validation_weight, params, show_plots=False):
            with torch.no_grad():
                y_pred_list = []
                for i_batch, (xb, yb, lens) in enumerate(test_dl):
                    xb = xb.cuda()
                    y_pred_list.append(model(xb, lens).detach().cpu())
                y_pred = torch.cat(y_pred_list)
            y_pred = {params.target_name: y_pred.detach().cpu().numpy()}
            y_pred = self.normalize_data(y_pred, self._data_scalar, 'inverse_transform', 'by_each_column')
            y_true = {params.target_name: y_validation}
            y_true = self.normalize_data(y_true, self._data_scalar, 'inverse_transform', 'by_each_column')
            all_scores = BaseModel.get_all_scores(y_true, y_pred, {params.target_name: params.fields},
                                                  validation_weight)
            all_scores = [{'subject': 'all', **scores} for scores in all_scores]
            self.print_table(all_scores)
            if show_plots:
                self.customized_analysis(y_true, y_pred, all_scores)
                plt.show()
            return y_pred

        def eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch):
            def vali_set_loss(nn_model, validation_dl, loss_fn):
                validation_loss = []
                for x_validation, y_validation, lens in validation_dl:
                    x_validation = x_validation.cuda()
                    y_validation = y_validation.cuda()
                    with torch.no_grad():
                        y_validation_pred = nn_model(x_validation, lens)
                        validation_loss.append(loss_fn(y_validation_pred, y_validation).item() / x_validation.shape[0])
                return np.mean(validation_loss)

            vali_from_train_loss = vali_set_loss(model, vali_from_train_dl, loss_fn)
            vali_from_test_loss = vali_set_loss(model, vali_from_test_dl, loss_fn)
            logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t".format(
                i_epoch, vali_from_train_loss, vali_from_test_loss, time.time() - epoch_end_time))

        params = SimpleNamespace(**params)
        train_dl, vali_from_train_dl, vali_from_test_dl, test_dl = prepare_sub_model_data(
            x_train, y_train, self.train_step_lens, x_validation, y_validation, self.validation_step_lens,
            params.batch_size)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        scheduler = ExponentialLR(optimizer, gamma=1)
        logging.info('\tEpoch | Vali_train_Loss | Vali_test_Loss | Duration\t\t')
        epoch_end_time = time.time()
        for i_epoch in range(params.epoch):
            eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch)
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, loss_fn, params)
            scheduler.step()
        eval_after_training(model, test_dl, y_validation, validation_weight, params)

    def predict(self, model, x_test):
        nn_model = model['four_source_model']
        model_fx_pre, model_fz_pre, model_rx_pre, model_rz_pre = model['model_fx_pre'], model['model_fz_pre'], model['model_rx_pre'], model['model_rz_pre']
        model_fx, model_fz, model_rx, model_rz = model['model_fx'], model['model_fz'], model['model_rx'], model['model_rz']
        self.test_step_lens = self._get_step_len(x_test)
        x_fx, x_fz, x_rx, x_rz, x_anthro = x_test['force_x'], x_test['force_z'], x_test['r_x'], x_test['r_z'], x_test[
            'anthro']
        x_fx, x_fz = torch.from_numpy(x_fx).float().cuda(), torch.from_numpy(x_fz).float().cuda()
        x_rx, x_rz, x_anthro = torch.from_numpy(x_rx).float().cuda(), torch.from_numpy(
            x_rz).float().cuda(), torch.from_numpy(x_anthro).float().cuda(),
        with torch.no_grad():
            test_ds = TensorDataset(x_fx, x_fz, x_rx, x_rz, x_anthro, torch.from_numpy(self.test_step_lens))
            test_dl = DataLoader(test_ds, batch_size=20)
            y_pred_list = []
            y_fx_pre, y_fz_pre, y_rx_pre, y_rz_pre = [], [], [], []
            y_fx, y_fz, y_rx, y_rz = [], [], [], []
            for i_batch, (xb_0, xb_1, xb_2, xb_3, xb_4, lens) in enumerate(test_dl):
                y_pred_list.append(nn_model(xb_0, xb_1, xb_2, xb_3, xb_4, lens).detach().cpu())
                y_fx_pre.append(model_fx_pre(xb_0, lens).detach().cpu())
                y_fz_pre.append(model_fz_pre(xb_1, lens).detach().cpu())
                y_rx_pre.append(model_rx_pre(xb_2, lens).detach().cpu())
                y_rz_pre.append(model_rz_pre(xb_3, lens).detach().cpu())
                y_fx.append(model_fx(xb_0, lens).detach().cpu())
                y_fz.append(model_fz(xb_1, lens).detach().cpu())
                y_rx.append(model_rx(xb_2, lens).detach().cpu())
                y_rz.append(model_rz(xb_3, lens).detach().cpu())
            y_pred = torch.cat(y_pred_list)
            y_fx_pre, y_fz_pre, y_rx_pre, y_rz_pre = torch.cat(y_fx_pre), torch.cat(y_fz_pre), torch.cat(y_rx_pre), torch.cat(y_rz_pre)
            y_fx, y_fz, y_rx, y_rz = torch.cat(y_fx), torch.cat(y_fz), torch.cat(y_rx), torch.cat(y_rz)
        y_pred = y_pred.detach().cpu().numpy()
        return {'main_output': y_pred,
                'midout_force_x': y_fx, 'midout_force_z': y_fz, 'midout_r_x': y_rx, 'midout_r_z': y_rz,
                'midout_force_x_pre': y_fx_pre, 'midout_force_z_pre': y_fz_pre, 'midout_r_x_pre': y_rx_pre, 'midout_r_z_pre': y_rz_pre}

    def save_temp_result(self, test_sub_y, pred_sub_y, test_sub_weight, models, test_sub_name):
        # save model
        save_path = os.path.join(self.result_dir, test_sub_name)
        os.mkdir(save_path)
        for model_name, model in models.items():
            torch.save(model.state_dict(), os.path.join(save_path, model_name + '.pth'))

        results = []
        columns = []
        for category, fields in self._y_fields.items():
            if len(fields) > 1:
                y_true_columns = fields
            else:
                y_true_columns = ['true_' + category]
            columns += y_true_columns
            results.append(test_sub_y[category])
        for category, fields_data in pred_sub_y.items():
            y_pred_columns = ['pred_' + category]
            columns += y_pred_columns
            results.append(fields_data)
        results = np.concatenate(results, axis=2)
        with h5py.File(os.path.join(self.result_dir, 'results.h5'), 'a') as hf:
            hf.create_dataset(test_sub_name, data=results, dtype='float32')
            hf.attrs['columns'] = json.dumps(columns)

    @staticmethod
    def _get_step_len(data, feature_col_num=0):
        """

        :param data: Numpy array, 3d (step, sample, feature)
        :param feature_col_num: int, feature column id for step length detection. Different id would probably return
               the same results
        :return:
        """
        data_the_feature = data[list(data.keys())[0]][:, :, feature_col_num]
        zero_loc = data_the_feature == 0.
        data_len = np.sum(~zero_loc, axis=1)
        return data_len

    def customized_analysis(self, sub_y_true, sub_y_pred, all_scores):
        """ Customized data visualization"""
        for score in all_scores:
            subject, output, field, r_rmse = [score[f] for f in ['subject', 'output', 'field', 'r_rmse']]
            arr1 = sub_y_true[output][:, :, 0]
            arr2 = sub_y_pred[output][:, :, 0]
            title = "{}, {}, {}, r_rmse".format(subject, output, field, 'r_rmse')
            self.representative_profile_curves(arr1, arr2, title, r_rmse)


def run(x_fields, y_fields, main_output_fields):
    weights = {key: [FORCE_PHASE] * len(y_fields[key]) for key in y_fields.keys()}
    weights.update({key: [FORCE_PHASE] * len(x_fields[key]) for key in x_fields.keys()})
    evaluate_fields = {'main_output': main_output_fields}
    model_builder = TianModel(data_path, x_fields, y_fields, weights, evaluate_fields,
                              lambda: MinMaxScaler(feature_range=(-3, 3)))
    subjects = model_builder.get_all_subjects()
    # model_builder.preprocess_train_evaluation(subjects[:13], subjects[13:], subjects[13:])
    model_builder.cross_validation(subjects)
    plt.show()


def run_kam(use_imu, use_op):
    input_imu = {'force_x': ACC_ML, 'force_z': ACC_VERTICAL, 'r_x': R_FOOT_SHANK_GYR, 'r_z': R_FOOT_SHANK_GYR}
    input_vid = {'force_x': VID_180_FIELDS, 'force_z': VID_180_FIELDS, 'r_x': VID_180_FIELDS, 'r_z': ['RKnee_y_90']}

    x_fields = {'force_x': [], 'force_z': [], 'r_x': [], 'r_z': []}
    if use_imu:
        x_fields = {k: x_fields[k] + input_imu[k] for k in list(x_fields.keys())}
    if use_op:
        x_fields = {k: x_fields[k] + input_vid[k] for k in list(x_fields.keys())}
    x_fields['anthro'] = STATIC_DATA

    main_output_fields = ['EXT_KM_Y']  # EXT_KM_Y RIGHT_KNEE_ADDUCTION_MOMENT
    y_fields = {
        'main_output': main_output_fields,
        'midout_force_x': ['plate_2_force_x'],
        'midout_force_z': ['plate_2_force_z'],
        'midout_r_x': ['KNEE_X'],
        'midout_r_z': ['KNEE_Z'],
        'auxiliary_info': [SUBJECT_ID, TRIAL_ID, FORCE_PHASE]
    }
    run(x_fields, y_fields, main_output_fields)


def run_kfm(use_imu, use_op):
    """ z -> y, x -> z"""
    input_imu = {'force_y': ACC_AP, 'force_z': ACC_VERTICAL, 'r_y': R_FOOT_SHANK_GYR, 'r_z': R_FOOT_SHANK_GYR}
    input_vid = {'force_y': VID_90_FIELDS, 'force_z': VID_90_FIELDS, 'r_y': VID_90_FIELDS, 'r_z': ['RKnee_y_90']}

    x_fields = {'force_y': [], 'force_z': [], 'r_y': [], 'r_z': []}
    if use_imu:
        x_fields = {k: x_fields[k] + input_imu[k] for k in list(x_fields.keys())}
    if use_op:
        x_fields = {k: x_fields[k] + input_vid[k] for k in list(x_fields.keys())}

    main_output_fields = ['EXT_KM_X']
    y_fields = {
        'main_output': main_output_fields,
        'midout_force_y': ['plate_2_force_y'],
        'midout_force_z': ['plate_2_force_z'],
        'midout_r_y': ['KNEE_Y'],
        'midout_r_z': ['KNEE_Z'],
        'auxiliary_info': [SUBJECT_ID, TRIAL_ID, FORCE_PHASE]
    }

    x_fields_renamed = {'force_x': x_fields['force_z'], 'force_z': x_fields['force_y'],
                        'r_x': x_fields['r_z'], 'r_z': x_fields['r_y'], 'anthro': STATIC_DATA}
    y_fields_renamed = {'midout_force_x': y_fields['midout_force_z'], 'midout_force_z': y_fields['midout_force_y'],
                        'midout_r_x': y_fields['midout_r_z'], 'midout_r_z': y_fields['midout_r_y'],
                        'main_output': y_fields['main_output'], 'auxiliary_info': y_fields['auxiliary_info']}
    run(x_fields_renamed, y_fields_renamed, main_output_fields)


if __name__ == "__main__":
    data_path = DATA_PATH + '/40samples+stance.h5'
    ACC_ML = ["AccelX_" + sensor for sensor in SENSOR_LIST]
    ACC_AP = ["AccelZ_" + sensor for sensor in SENSOR_LIST]
    ACC_VERTICAL = ["AccelY_" + sensor for sensor in SENSOR_LIST]
    VID_180_FIELDS = [loc + axis + '_180' for loc in ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
                      for axis in ['_x', '_y']]
    VID_90_FIELDS = [loc + axis + '_90' for loc in ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"] for axis in ['_x', '_y']]
    VID_90_R_LIMB_FIELDS = [loc + axis + '_90' for loc in ["RShoulder", "RKnee", "RAnkle"] for axis in ['_x', '_y']]
    R_FOOT_SHANK_GYR = ["Gyro" + axis + sensor for sensor in ['R_SHANK', 'R_FOOT'] for axis in ['X_', 'Y_', 'Z_']]

    run_kam(use_imu=True, use_op=True)
    run_kam(use_imu=False, use_op=True)
    run_kam(use_imu=True, use_op=False)

    run_kfm(use_imu=True, use_op=True)
    run_kfm(use_imu=False, use_op=True)
    run_kfm(use_imu=True, use_op=False)
