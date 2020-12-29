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
from const import IMU_FIELDS, SENSOR_LIST, DATA_PATH, VIDEO_LIST, SUBJECT_WEIGHT, SUBJECT_HEIGHT, FORCE_PHASE, \
    FORCE_DATA_FIELDS, STATIC_DATA, SEGMENT_MASS_PERCENT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MinMaxScaler
from types import SimpleNamespace
import pandas as pd

USE_GPU = True


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
        self.conv2output = nn.Linear(64, y_dim * 100, bias=False)
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
        output = torch.reshape(output, (-1, 100, self.y_dim))
        return output


class TianRNN(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=10, nlayer=2):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.i2r = nn.Linear(x_dim, hidden_dim)
        self.rnn_layer = nn.LSTM(hidden_dim, hidden_dim, nlayer, batch_first=True, bidirectional=True)
        self.y_dim = y_dim
        self.r2d = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.d2o = nn.Linear(hidden_dim, y_dim, bias=False)
        self.relu = nn.ReLU()
        for layer in [self.i2r, self.r2d, self.d2o]:
            nn.init.xavier_normal_(layer.weight)
        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sequence, lens):
        sequence = self.i2r(sequence)
        sequence = self.relu(sequence)
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.rnn_layer(sequence)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=152)
        sequence = self.r2d(lstm_out)
        sequence = self.relu(sequence)
        output = self.d2o(sequence)
        return output


class FCModel(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(FCModel, self).__init__()
        hidden_num = 20
        self.linear1 = nn.Linear(x_dim, hidden_num, bias=False)
        self.linear2 = nn.Linear(hidden_num, hidden_num, bias=False)
        self.linear3 = nn.Linear(hidden_num, hidden_num, bias=False)
        self.linear4 = nn.Linear(hidden_num, hidden_num, bias=False)
        self.hidden2output = nn.Linear(hidden_num, y_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequence, lens):
        sequence = self.sigmoid(self.linear1(sequence))
        sequence = self.sigmoid(self.linear3(sequence))
        # sequence = self.sigmoid(self.linear4(sequence))

        # sequence = self.linear1(sequence).clamp(min=0)
        # sequence = self.linear2(sequence).clamp(min=0)
        # sequence = self.linear3(sequence).clamp(min=0)
        # sequence = self.linear4(sequence).clamp(min=0)
        sequence = self.hidden2output(sequence)
        return sequence


class TianLinear(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(TianLinear, self).__init__()
        self.linear = nn.Linear(x_dim, y_dim)

    def forward(self, sequence, lens=None):
        sequence = self.linear(sequence)
        return sequence


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
        min_, scale_ = self.scalars[fields].min_[0], self.scalars[fields].scale_[0]
        data[data == 0.] = np.nan
        data = torch.add(data, - min_)
        data = torch.div(data, scale_)
        data[torch.isnan(data)] = 0.
        return data


class TianModel(BaseModel):
    def __init__(self, data_path, x_fields, y_fields, weights, base_scalar):
        BaseModel.__init__(self, data_path, x_fields, y_fields, weights, base_scalar)
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3
        self.vid_static_cali()
        self.add_additional_columns()

    def vid_static_cali(self):
        vid_y_90_col_loc = [self._data_fields.index(marker + '_y_90') for marker in VIDEO_LIST]
        for sub_name, sub_data in self._data_all_sub.items():
            static_side_df = pd.read_csv(DATA_PATH + '/' + sub_name + '/combined/static_side.csv', index_col=0)
            r_ankle_z = np.median(static_side_df['RAnkle_y_90'])
            sub_data[:, :, vid_y_90_col_loc] = sub_data[:, :, vid_y_90_col_loc] - r_ankle_z + 1500
            self._data_all_sub[sub_name] = sub_data

    def add_additional_columns(self):
        marker_col_loc = [self._data_fields.index(field_name) for field_name in KNEE_MARKER_FIELDS]
        force_col_loc = [self._data_fields.index(field_name) for field_name in FORCE_DATA_FIELDS]
        # vid_col_loc = [self._data_fields.index(marker + '_x_180') for marker in ["LShoulder", "RShoulder", "MidHip"]]
        # plt.figure()
        for sub_name, sub_data in self._data_all_sub.items():
            marker_data = sub_data[:, :, marker_col_loc].copy()
            force_data = sub_data[:, :, force_col_loc].copy()
            knee_vector = force_data[:, :, 9:12] - (marker_data[:, :, :3] + marker_data[:, :, 3:6]) / 2
            total_force = force_data[:, :, :3] + force_data[:, :, 6:9]
            self._data_all_sub[sub_name] = np.concatenate([sub_data, knee_vector, total_force], axis=2)
        #     vid_data = sub_data[:, :, vid_col_loc]
        #     trunk_sway_x = (vid_data[:, :, 0] + vid_data[:, :, 1]) / 2 - vid_data[:, :, 2]
        #     plt.plot(force_data[:, :, 9].ravel(), trunk_sway_x.ravel(), '.')
        # plt.show()
        self._data_fields.extend(['KNEE_X', 'KNEE_Y', 'KNEE_Z', 'TOTAL_F_X', 'TOTAL_F_Y', 'TOTAL_F_Z'])

    def preprocess_train_data(self, x, y, weight):
        x['midout_force_x'], x['midout_force_z'] = -x['midout_force_x'], -x['midout_force_z']
        x['midout_r_x'], x['midout_r_z'] = x['midout_r_x'] / 1000, x['midout_r_z'] / 1000
        self.x_need_norm = {k: x[k] for k in set(list(x.keys())) - set(['anthro'])}
        x.update(**self.normalize_data(self.x_need_norm, self._data_scalar, 'fit_transform', scalar_mode='by_each_column'))

        return x, y, weight

    def preprocess_validation_test_data(self, x, y, weight):
        x['midout_force_x'], x['midout_force_z'] = -x['midout_force_x'], -x['midout_force_z']
        x['midout_r_x'], x['midout_r_z'] = x['midout_r_x'] / 1000, x['midout_r_z'] / 1000
        self.x_need_norm = {k: x[k] for k in set(list(x.keys())) - set(['anthro'])}
        x.update(**self.normalize_data(self.x_need_norm, self._data_scalar, 'transform', scalar_mode='by_each_column'))
        return x, y, weight

    @staticmethod
    def loss_fun_emphasize_peak(y_pred, y):
        peak_locs = torch.argmax(y, dim=1).reshape([-1])
        loss_peak = (y_pred[range(y.shape[0]), peak_locs] - y[range(y.shape[0]), peak_locs]).pow(2).mean()
        y_pred_non_zero = y_pred[y != 0]
        y_non_zero = y[y != 0]
        loss_profile = (y_pred_non_zero - y_non_zero).pow(2).mean()
        return (loss_profile + loss_peak) * 1e3

    @staticmethod
    def loss_fun_only_positive(y_pred, y):
        y_pred_positive = y_pred[y > 0]
        y_positive = y[y > 0]
        loss_positive = (y_pred_positive - y_positive).pow(2).sum()
        return loss_positive

    @staticmethod
    def loss_fun_valid_part(y_pred, y, left, right):
        y_pred_valid = y_pred[:, left:-right]
        y_valid = y[:, left:-right]
        loss_positive = (y_pred_valid - y_valid).pow(2).sum()
        return loss_positive

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        sub_model_base_param = {'epoch': 12, 'batch_size': 20, 'lr': 1e-3, 'weight_decay': 1e-5, 'use_ratio': 20}
        self.train_step_lens, self.validation_step_lens = self._get_step_len(x_train), self._get_step_len(x_validation)

        x_train_rx, x_validation_rx = x_train['r_x'], x_validation['r_x']
        y_train_rx, y_validation_rx = x_train['midout_r_x'], x_validation['midout_r_x']
        model_rx = TianRNN(x_train_rx.shape[2], y_train_rx.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_r_x', 'fields': ['KNEE_X']}}
        self.build_sub_model(model_rx, x_train_rx, y_train_rx, x_validation_rx, y_validation_rx, validation_weight, params)

        x_train_rz, x_validation_rz = x_train['r_z'], x_validation['r_z']
        y_train_rz, y_validation_rz = x_train['midout_r_z'], x_validation['midout_r_z']
        model_rz = TianRNN(x_train_rz.shape[2], y_train_rz.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_r_z', 'fields': ['KNEE_Z']}}
        self.build_sub_model(model_rz, x_train_rz, y_train_rz, x_validation_rz, y_validation_rz, validation_weight, params)

        x_train_fz, x_validation_fz = x_train['force_z'], x_validation['force_z']
        y_train_fz, y_validation_fz = x_train['midout_force_z'], x_validation['midout_force_z']
        model_fz = TianRNN(x_train_fz.shape[2], y_train_fz.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_force_z', 'fields': ['plate_2_force_z']}}
        self.build_sub_model(model_fz, x_train_fz, y_train_fz, x_validation_fz, y_validation_fz, validation_weight, params)

        x_train_fx, x_validation_fx = x_train['force_x'], x_validation['force_x']
        y_train_fx, y_validation_fx = x_train['midout_force_x'], x_validation['midout_force_x']
        model_fx = TianRNN(x_train_fx.shape[2], y_train_fx.shape[2]).cuda()
        params = {**sub_model_base_param, **{'target_name': 'midout_force_x', 'fields': ['plate_2_force_x']}}
        self.build_sub_model(model_fx, x_train_fx, y_train_fx, x_validation_fx, y_validation_fx, validation_weight, params)

        four_source_model = FourSourceModel(model_fx, model_fz, model_rx, model_rz, self._data_scalar).cuda()
        params = {**sub_model_base_param, **{'target_name': 'main_output', 'fields': ['EXT_KM_Y']}}
        params['lr'] = params['lr'] * 0.1
        params['batch_size'] = 200
        self.build_main_model(four_source_model, x_train, y_train, x_validation, y_validation, validation_weight, params)
        return four_source_model

    def build_main_model(self, model, x_train, y_train, x_validation, y_validation, validation_weight, params):
        def prepare_main_model_data(x_train, y_train, train_step_lens, x_validation, y_validation, validation_step_lens,
                                    batch_size):
            x_fx, x_fz, x_rx, x_rz, x_anthro = x_train['force_x'], x_train['force_z'], x_train['r_x'], x_train['r_z'], x_train['anthro']
            y_train = torch.from_numpy(y_train['main_output']).float().cuda()
            x_fx, x_fz = torch.from_numpy(x_fx).float().cuda(), torch.from_numpy(x_fz).float().cuda()
            x_rx, x_rz, x_anthro = torch.from_numpy(x_rx).float().cuda(), torch.from_numpy(x_rz).float().cuda(), torch.from_numpy(x_anthro).float().cuda()
            train_step_lens = torch.from_numpy(train_step_lens)
            train_ds = TensorDataset(x_fx, x_fz, x_rx, x_rz, x_anthro, y_train, train_step_lens)
            train_size = int(0.9 * len(train_ds))
            vali_from_train_size = len(train_ds) - train_size
            train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds, [train_size, vali_from_train_size])
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=batch_size)

            x_fx, x_fz, x_rx, x_rz, x_anthro = x_validation['force_x'], x_validation['force_z'], x_validation['r_x'], x_validation['r_z'], x_validation['anthro']
            y_validation = torch.from_numpy(y_validation['main_output']).float().cuda()
            x_fx, x_fz = torch.from_numpy(x_fx).float().cuda(), torch.from_numpy(x_fz).float().cuda()
            x_rx, x_rz, x_anthro = torch.from_numpy(x_rx).float().cuda(), torch.from_numpy(x_rz).float().cuda(), torch.from_numpy(x_anthro).float().cuda()
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
                # train_loss = self.loss_fun_emphasize_peak(y_pred, yb)
                # train_loss = self.loss_fun_only_positive(y_pred, yb)
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
            all_scores = BaseModel.get_all_scores(y_validation, y_pred, {params.target_name: params.fields}, validation_weight)
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
                # train_loss = self.loss_fun_emphasize_peak(y_pred, yb)
                # train_loss = self.loss_fun_only_positive(y_pred, yb)
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
            all_scores = BaseModel.get_all_scores(y_true, y_pred, {params.target_name: params.fields}, validation_weight)
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
                # plt.figure()
                # plt.plot(x_validation[0, :, 0].cpu().numpy())
                # plt.plot(y_validation_pred[0, :, 0].cpu().numpy())
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

    def predict(self, nn_model, x_test):
        self.test_step_lens = self._get_step_len(x_test)
        x_fx, x_fz, x_rx, x_rz, x_anthro = x_test['force_x'], x_test['force_z'], x_test['r_x'], x_test['r_z'], x_test['anthro']
        x_fx, x_fz = torch.from_numpy(x_fx).float().cuda(), torch.from_numpy(x_fz).float().cuda()
        x_rx, x_rz, x_anthro = torch.from_numpy(x_rx).float().cuda(), torch.from_numpy(x_rz).float().cuda(), torch.from_numpy(x_anthro).float().cuda(),
        with torch.no_grad():
            test_ds = TensorDataset(x_fx, x_fz, x_rx, x_rz, x_anthro, torch.from_numpy(self.test_step_lens))
            test_dl = DataLoader(test_ds, batch_size=20)
            y_pred_list = []
            for i_batch, (xb_0, xb_1, xb_2, xb_3, xb_4, lens) in enumerate(test_dl):
                y_pred_list.append(nn_model(xb_0, xb_1, xb_2, xb_3, xb_4, lens).detach().cpu())
            y_pred = torch.cat(y_pred_list)
        y_pred = y_pred.detach().cpu().numpy()
        self._y_fields = {'main_output': self._y_fields['main_output']}
        return {'main_output': y_pred}

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


if __name__ == "__main__":
    data_path = DATA_PATH + '/40samples+stance.h5'
    IMU_FIELDS_ACC = IMU_FIELDS[:3]
    IMU_FIELDS_GYR = IMU_FIELDS[3:6]
    IMU_FIELDS_ORI = IMU_FIELDS[9:13]
    IMU_DATA_FIELDS_ACC = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_ACC]
    IMU_DATA_FIELDS_GYR = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_GYR]

    R_LEG_ACC = [imu_field + "_" + sensor for sensor in ['R_SHANK', 'R_FOOT'] for imu_field in IMU_FIELDS_ACC]
    R_LEG_GYR = [imu_field + "_" + sensor for sensor in ['R_SHANK', 'R_FOOT'] for imu_field in IMU_FIELDS_GYR]
    R_LEG_ORI = [ori_field + "_" + sensor for sensor in ['R_SHANK', 'R_FOOT'] for ori_field in IMU_FIELDS_ORI]

    ACC_VERTICAL = ["AccelY_" + sensor for sensor in SENSOR_LIST[2:]]
    ACC_ML = ["AccelX_" + sensor for sensor in SENSOR_LIST[2:]]
    GYR_ML = ["GyroX_" + sensor for sensor in SENSOR_LIST[2:]]

    MARKER_Z = [marker + '_Z' for marker in ['RFME', 'RFLE', 'RTAM', 'RFAL']]
    MARKER_X = [marker + '_X' for marker in ['RFME', 'RFLE', 'RTAM', 'RFAL']]

    KNEE_MARKER_FIELDS = [marker + axis for marker in ['RFME', 'RFLE'] for axis in ['_X', '_Y', '_Z']]
    ANKLE_MARKER_FIELDS = [marker + axis for marker in ['RTAM', 'RFAL'] for axis in ['_X', '_Y', '_Z']]
    TRUNK_MARKER_FIELDS = [marker + axis for marker in ['CV7'] for axis in ['_X', '_Y', '_Z']]

    VID_FIELDS = [loc + '_' + axis + '_' + angle for loc in ['RShoulder', 'RHip'] for angle in ['180'] for axis in ['x']]
    x_fields = {
        'force_x': ACC_ML + GYR_ML,
        'force_z': ACC_VERTICAL,
        'r_x': [axis + sensor for axis in ["AccelX_", 'GyroZ_'] for sensor in ['WAIST', 'CHEST']],            # GyroZ_
        'r_z': ['RKnee_y_90', 'GyroX_R_FOOT'],
        # 'r_x': MARKER_X,            # GyroZ_
        # 'r_z': MARKER_Z,
        'anthro': STATIC_DATA,
        'midout_force_x': ['plate_2_force_x'],
        'midout_force_z': ['plate_2_force_z'],
        'midout_r_x': ['KNEE_X'],
        'midout_r_z': ['KNEE_Z'],
    }
    MAIN_OUTPUT_FIELDS = ['EXT_KM_Y']  # EXT_KM_Y RIGHT_KNEE_ADDUCTION_MOMENT
    y_fields = {
        'main_output': MAIN_OUTPUT_FIELDS
    }

    weights = {key: [FORCE_PHASE] * len(y_fields[key]) for key in y_fields.keys()}
    weights.update({key: [FORCE_PHASE] * len(x_fields[key]) for key in x_fields.keys()})
    model_builder = TianModel(data_path, x_fields, y_fields, weights, lambda: MinMaxScaler(feature_range=(0, 3)))
    subjects = model_builder.get_all_subjects()
    # model_builder.preprocess_train_evaluation(subjects[:13], subjects[13:], subjects[13:])
    model_builder.cross_validation(subjects)
    plt.show()
