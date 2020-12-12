import random

from base_kam_model import BaseModel
from wearable_toolkit import DivideMaxScalar
import torch
import torch.nn as nn
from customized_logger import logger as logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np
import time
from const import IMU_FIELDS, SENSOR_LIST, DATA_PATH, KAM_PHASE, SUBJECT_WEIGHT, SUBJECT_HEIGHT, FORCE_PHASE, \
    FORCE_DATA_FIELDS, STATIC_DATA
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MinMaxScaler

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

    def forward(self, sequence):
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
    def __init__(self, x_dim, y_dim, hidden_dim=15, nlayer=2):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.rnn_layer = nn.LSTM(x_dim, hidden_dim, nlayer, dropout=0, batch_first=True, bidirectional=True, bias=False)
        self.y_dim = y_dim
        self.hidden2output = nn.Linear(2 * hidden_dim, y_dim, bias=False)
        # self.hidden2dense = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.dense2output = nn.Linear(hidden_dim, y_dim)
        # for layer in [self.hidden2dense, self.dense2output]:
        for layer in [self.hidden2output]:
            nn.init.xavier_normal_(layer.weight)
        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sequence, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.rnn_layer(sequence)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=231)
        output = self.hidden2output(lstm_out)
        # relu_out = self.hidden2dense(lstm_out).clamp(min=0)
        # output = self.dense2output(relu_out)
        return output


# class CrossProductModel(nn.Module):
#     def __init__(self):
#         super(CrossProductModel, self).__init__()


class TianModel(BaseModel):
    def __init__(self, data_path, x_fields, y_fields, weights, base_scalar):
        BaseModel.__init__(self, data_path, x_fields, y_fields, weights, base_scalar)
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3

    def preprocess_train_data(self, x, y, weight):
        # x['main_input_acc'] = x['main_input_acc'] / 10

        # marker_data, force_data = x['main_input_marker'], x['main_input_force']
        # knee_vector = force_data[:, :, 3:6] - (marker_data[:, :, :3] + marker_data[:, :, 3:6]) / 2
        # kam = (knee_vector[:, :, 2] * force_data[:, :, 0] - knee_vector[:, :, 0] * force_data[:, :, 2]) / (
        #             1000 * x['main_input_acc'][:, :, 0] * x['main_input_acc'][:, :, 1])
        # plt.figure()
        # plt.plot(-kam[:100, :].ravel())
        # plt.plot(y['main_output'][:100, :, 0].ravel())
        # plt.show()

        x = self.normalize_data(x, self._data_scalar, 'fit_transform')
        # y['output'] = self.resample_stance_phase_kam(y['output'], weight['output'])
        return x, y, weight

    def preprocess_validation_test_data(self, x, y, weight):
        self.preprocess_train_data(x, y, weight)
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
        self.train_step_lens, self.validation_step_lens = self._get_step_len(x_train), self._get_step_len(x_validation)
        x_train, y_train = np.concatenate(list(x_train.values()), axis=2), y_train['main_output']
        x_validation, y_validation = np.concatenate(list(x_validation.values()), axis=2), y_validation['main_output']

        # x_train, y_train = x_train[:5000], y_train[:5000]

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        train_step_lens = torch.from_numpy(self.train_step_lens)
        # nn_model = TianCNN(14, 1)
        nn_model = TianRNN(14, 1)

        if USE_GPU:
            nn_model = nn_model.cuda()
        # summary(nn_model, x_train.shape[1:])
        pytorch_total_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
        logging.info('Model has {} parameters.'.format(pytorch_total_params))

        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=5e-4, weight_decay=2e-6)
        # optimizer = torch.optim.Adam(nn_model.parameters())

        batch_size = 20
        train_ds = TensorDataset(x_train, y_train, train_step_lens)
        train_size = int(0.9 * len(train_ds))
        vali_from_train_size = len(train_ds) - train_size
        train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds,
                                                                             [train_size, vali_from_train_size])
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=batch_size)

        x_validation = torch.from_numpy(x_validation).float()
        y_validation = torch.from_numpy(y_validation).float()
        vali_step_lens = torch.from_numpy(self.validation_step_lens)
        vali_from_test_ds = TensorDataset(x_validation, y_validation, vali_step_lens)
        num_of_step_for_peek = int(0.1 * len(x_validation))
        vali_from_test_ds, _ = torch.utils.data.dataset.random_split(vali_from_test_ds, [num_of_step_for_peek, len(
            x_validation) - num_of_step_for_peek])
        vali_from_test_dl = DataLoader(vali_from_test_ds, batch_size=batch_size)

        logging.info('\tEpoch\t\tTrain_Loss\tVali_train_Loss\tVali_test_Loss\t\tDuration\t\t')
        for epoch in range(20):
            epoch_start_time = time.time()
            for i_batch, (xb, yb, lens) in enumerate(train_dl):
                if i_batch > 1:
                    n = random.randint(1, 100)
                    if n > 10:
                        continue  # increase the speed of epoch

                if USE_GPU:
                    xb = xb.cuda()
                    yb = yb.cuda()

                optimizer.zero_grad()

                y_pred = nn_model(xb, lens)

                # train_loss = self.loss_fun_emphasize_peak(y_pred, yb)
                # train_loss = self.loss_fun_only_positive(y_pred, yb)
                train_loss = loss_fn(y_pred, yb)

                if i_batch == 0:
                    vali_from_train_loss = TianModel.evaluate_validation_set(nn_model, vali_from_train_dl, loss_fn)
                    vali_from_test_loss = TianModel.evaluate_validation_set(nn_model, vali_from_test_dl, loss_fn)
                    logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t\t".format(
                        epoch, train_loss.item() / xb.shape[0], vali_from_train_loss, vali_from_test_loss,
                               time.time() - epoch_start_time))
                train_loss.backward()
                optimizer.step()

            # vali_from_train_loss = TianModel.evaluate_validation_set(nn_model, vali_from_train_dl, loss_fn)
            # vali_from_test_loss = TianModel.evaluate_validation_set(nn_model, vali_from_test_dl, loss_fn)
            # logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t\t".format(
            #     epoch, train_loss.item() / xb.shape[0], vali_from_train_loss, vali_from_test_loss,
            #     time.time() - epoch_start_time))
        return nn_model

    @staticmethod
    def evaluate_validation_set(nn_model, validation_dl, loss_fn):
        validation_loss = []
        for x_validation, y_validation, lens in validation_dl:
            if USE_GPU:
                x_validation = x_validation.cuda()
                y_validation = y_validation.cuda()

            with torch.no_grad():
                # For RNN
                y_validation_pred = nn_model(x_validation, lens)

                validation_loss.append(loss_fn(y_validation_pred, y_validation).item() / x_validation.shape[0])
        return np.mean(validation_loss)

    def predict(self, nn_model, x_test):
        self.test_step_lens = self._get_step_len(x_test)
        x_test = np.concatenate(list(x_test.values()), axis=2)
        x_test = torch.from_numpy(x_test).float()
        if USE_GPU:
            x_test = x_test.cuda()
        with torch.no_grad():
            # # For RNN
            # hidden = nn_model.init_hidden(x_test.shape[0])
            # y_pred, _ = nn_model(x_test, hidden, self.test_step_lens)

            # x_test = torch.zeros(x_test.shape, device='cuda')

            test_ds = TensorDataset(x_test, torch.from_numpy(self.test_step_lens))
            test_dl = DataLoader(test_ds, batch_size=20)
            y_pred_list = []
            for i_batch, (xb, lens) in enumerate(test_dl):
                y_pred_list.append(nn_model(xb, lens).detach().cpu())
            y_pred = torch.cat(y_pred_list)
        y_pred = y_pred.detach().cpu().numpy()
        return {'main_output': y_pred}

    @staticmethod
    def _get_step_len(data, input_cate='main_input_acc', feature_col_num=0):
        """

        :param data: Numpy array, 3d (step, sample, feature)
        :param feature_col_num: int, feature column id for step length detection. Different id would probably return
               the same results
        :return:
        """
        data_the_feature = data[input_cate][:, :, feature_col_num]
        zero_loc = data_the_feature == 0.
        data_len = np.sum(~zero_loc, axis=1)
        return data_len


if __name__ == "__main__":
    data_path = DATA_PATH + '/40samples+stance_swing+padding_zero.h5'
    IMU_FIELDS_ACC = IMU_FIELDS[:3]
    IMU_FIELDS_GYR = IMU_FIELDS[3:6]
    IMU_FIELDS_ORI = IMU_FIELDS[9:13]
    # IMU_DATA_FIELDS_ACC = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_ACC]
    # IMU_DATA_FIELDS_GYR = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_GYR]
    video_cols = [loc + '_' + axis + '_' + angle for loc in ['RHip', 'LHip', 'RKnee', 'LKnee']
                  for angle in ['90', '180'] for axis in ['x', 'y']]
    output_cols = ['RIGHT_KNEE_ADDUCTION_MOMENT', 'EXT_KM_Y']

    R_LEG_ACC = [imu_field + "_" + sensor for sensor in ['R_SHANK', 'R_THIGH'] for imu_field in IMU_FIELDS_ACC]
    R_LEG_GYR = [imu_field + "_" + sensor for sensor in ['R_SHANK', 'R_THIGH'] for imu_field in IMU_FIELDS_ACC]
    R_LEG_ORI = [ori_field + "_" + sensor for sensor in ['R_SHANK', 'R_THIGH'] for ori_field in IMU_FIELDS_ORI]

    MARKER_FIELDS = [marker + axis for marker in ['RFME', 'RFLE'] for axis in ['_X', '_Y', '_Z']]

    x_fields = {
        'main_input_acc': R_LEG_ACC,
        'main_input_gyr': R_LEG_GYR,
        # 'main_input_ori': R_LEG_ORI,
        'main_input_anthro': STATIC_DATA,
        # 'main_input_vid': video_cols,
        # 'aux_input': [SUBJECT_WEIGHT, SUBJECT_HEIGHT]
    }
    MAIN_OUTPUT_FIELDS = ['EXT_KM_Y']  # EXT_KM_Y RIGHT_KNEE_ADDUCTION_MOMENT
    y_fields = {'main_output': MAIN_OUTPUT_FIELDS,
                'mid_output_marker': MARKER_FIELDS,
                'mid_output_force': FORCE_DATA_FIELDS[6:12]}

    weights = {'main_output': [FORCE_PHASE] * len(output_cols)}
    model = TianModel(data_path, x_fields, y_fields, weights, lambda: MinMaxScaler(feature_range=(0, 1)))
    subjects = model.get_all_subjects()
    model.preprocess_train_evaluation(subjects[:13], subjects[13:], subjects[13:])
    # model.cross_validation(subjects)
    plt.show()
