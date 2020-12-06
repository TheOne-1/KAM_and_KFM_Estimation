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
from const import IMU_FIELDS, SENSOR_LIST, DATA_PATH, KAM_PHASE, SUBJECT_WEIGHT, SUBJECT_HEIGHT, FORCE_PHASE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torchsummary import summary
import torch.nn.functional as F
import scipy.interpolate as interpo

USE_GPU = True


class TianCNN1(nn.Module):
    """Large Scale CNN"""

    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pooling2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pooling3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.pooling4 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.conv2fc = nn.Linear(448, 100, bias=False)
        self.fc2output = nn.Linear(100, y_dim * 50, bias=False)
        self.y_dim = y_dim

    def forward(self, sequence):
        sequence = sequence[:, 30:100, :]       # take part of the data
        sequence = sequence.unsqueeze(1)
        sequence = self.relu(self.conv1(sequence))
        # sequence = self.bn1(sequence)
        sequence = self.pooling1(sequence)
        sequence = self.relu(self.conv2(sequence))
        # sequence = self.bn2(sequence)
        sequence = self.pooling2(sequence)
        sequence = self.relu(self.conv3(sequence))
        # sequence = self.bn3(sequence)
        sequence = self.pooling3(sequence)
        # sequence = self.relu(self.conv4(sequence))
        # sequence = self.bn4(sequence)
        # sequence = self.pooling4(sequence)
        sequence = self.flatten(sequence)
        output = self.relu(self.conv2fc(sequence))
        output = self.fc2output(output)
        output = torch.reshape(output, (-1, 50, self.y_dim))
        return output


class TianCNN2(nn.Module):
    """time 1d CNN, then channel 1d CNN"""
    def __init__(self, x_dim, y_dim):
        super().__init__()
        kernel_num_1 = 8
        self.conv1 = [nn.Conv1d(1, 2 * kernel_num_1, kernel_size=9, stride=3, bias=False).cuda() for _ in range(x_dim)]
        self.conv2 = [nn.Conv1d(2 * kernel_num_1, kernel_num_1, kernel_size=9, stride=3, bias=False).cuda() for _ in
                      range(x_dim)]
        self.conv3 = [nn.Conv1d(kernel_num_1, kernel_num_1, kernel_size=9, stride=3, bias=False).cuda() for _ in
                      range(x_dim)]

        kernel_num_2 = 8
        self.conv5 = nn.Conv1d(5, kernel_num_2, kernel_size=9, stride=3, bias=False)
        self.conv6 = nn.Conv1d(kernel_num_2, kernel_num_2, kernel_size=9, stride=3, bias=False)
        self.conv7 = nn.Conv1d(kernel_num_2, kernel_num_2, kernel_size=9, stride=3, bias=False)
        self.conv8 = nn.Conv1d(kernel_num_2, kernel_num_2, kernel_size=9, stride=3, bias=False)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.conv2output = nn.Linear(88, y_dim * 50, bias=False)
        self.y_dim = y_dim
        self.x_dim = x_dim

    def forward(self, sequence):
        feature_outputs = []
        sequence.transpose_(1, 2)
        for i_feature in range(self.x_dim):
            narrowed = sequence.narrow(1, i_feature, 1)
            narrowed = self.conv1[i_feature](narrowed)
            narrowed = self.conv2[i_feature](narrowed)
            narrowed = self.conv3[i_feature](narrowed)
            feature_outputs.append(narrowed)
        feature_outputs = torch.cat(feature_outputs, dim=1)
        feature_outputs = feature_outputs.transpose(1, 2)

        feature_fused = self.conv5(feature_outputs)
        feature_fused = self.conv6(feature_fused)
        feature_fused = self.conv7(feature_fused)
        sequence = self.flatten(feature_fused)
        output = self.conv2output(sequence)
        output = torch.reshape(output, (-1, 50, self.y_dim))
        return output


class TianCNN3(nn.Module):
    """Small and quick"""

    def __init__(self, x_dim, y_dim):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 256, kernel_size=(30, 1), stride=1, bias=False)
        self.kernel_num = 1
        self.x_dim = x_dim
        self.conv1 = [nn.Conv1d(1, self.kernel_num, kernel_size=50, stride=1, bias=False).cuda() for _ in range(x_dim)]
        self.bn1 = [nn.BatchNorm1d(1, self.kernel_num).cuda() for _ in range(x_dim)]
        self.flatten = nn.Flatten()
        self.conv2fc = nn.Linear(self.kernel_num*x_dim, y_dim * 50, bias=False)
        self.y_dim = y_dim
        for layer in self.conv1:
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, sequence):
        # sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        sequence = sequence[:, 30:100, :]       # take part of the data
        sequence.transpose_(1, 2)
        feature_outputs = []
        for i_feature in range(self.x_dim):
            narrowed = sequence.narrow(1, i_feature, 1)
            narrowed = self.conv1[i_feature](narrowed)
            narrowed = F.relu(narrowed)
            feature_output, _ = torch.max(narrowed, 2)
            # feature_output = self.bn1[i_feature](feature_output)
            feature_outputs.append(feature_output)
        feature_outputs = torch.cat(feature_outputs, dim=1)
        sequence = self.flatten(feature_outputs)
        output = self.conv2fc(sequence)
        output = torch.reshape(output, (-1, 50, self.y_dim))
        return output


class TianCNN4(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        kernel_num = 32
        self.conv1 = nn.Conv1d(x_dim, 8 * kernel_num, kernel_size=3, stride=1, bias=False)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8 * kernel_num, 2 * kernel_num, kernel_size=3, stride=1, bias=False)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(2 * kernel_num, kernel_num, kernel_size=3, stride=1, bias=False)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(kernel_num, kernel_num, kernel_size=3, stride=1, bias=False)
        self.pool4 = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.conv2output = nn.Linear(64, y_dim * 100, bias=False)
        self.drop = nn.Dropout(p=0.2)
        self.y_dim = y_dim
        self.x_dim = x_dim

    def forward(self, sequence):
        sequence = sequence[:, 30:100, :]       # take part of the data
        sequence.transpose_(1, 2)
        sequence = self.relu(self.conv1(sequence))
        sequence = self.drop(sequence)
        sequence = self.pool1(sequence)
        sequence = self.relu(self.conv2(sequence))
        sequence = self.drop(sequence)
        sequence = self.pool2(sequence)
        sequence = self.relu(self.conv3(sequence))
        sequence = self.drop(sequence)
        sequence = self.pool3(sequence)
        sequence = self.relu(self.conv4(sequence))
        sequence = self.drop(sequence)
        sequence = self.pool4(sequence)
        sequence = self.flatten(sequence)
        output = self.conv2output(sequence)
        output = torch.reshape(output, (-1, 100, self.y_dim))
        return output


class TianRNN(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=30, nlayer=2):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.rnn_layer = nn.LSTM(x_dim, hidden_dim, nlayer, dropout=0, batch_first=True, bidirectional=True)
        self.y_dim = y_dim
        self.hidden2dense = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dense2output = nn.Linear(hidden_dim, y_dim)
        for layer in [self.hidden2dense, self.dense2output]:
            nn.init.xavier_normal_(layer.weight)
        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sequence, hidden, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.rnn_layer(sequence, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=230)
        relu_out = self.hidden2dense(lstm_out).clamp(min=0)
        output = self.dense2output(relu_out)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayer * 2, batch_size, self.hidden_dim),
                weight.new_zeros(self.nlayer * 2, batch_size, self.hidden_dim))


class TianCLDNN(nn.Module):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super(TianCLDNN, self).__init__()
        kernel_num = 16
        self.kernel_num = kernel_num
        self.conv1 = nn.Conv2d(1, 2 * kernel_num, kernel_size=(8, 1), stride=(3, 1), bias=False)
        # self.bn1 = nn.BatchNorm2d(3*kernel_num)
        self.conv2 = nn.Conv2d(2 * kernel_num, kernel_num, kernel_size=(8, 1), stride=(3, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(kernel_num)
        # self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(kernel_num, kernel_num, kernel_size=(1, 9), stride=(1, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(kernel_num)
        # self.pool3 = nn.MaxPool2d(kernel_size=(23, 13))
        self.conv4 = nn.Conv2d(kernel_num, kernel_num, kernel_size=(1, 6), stride=(1, 2), bias=False)
        self.bn4 = nn.BatchNorm2d(kernel_num)
        self.cnn2rnn = nn.Linear(80, kernel_num)
        self.flatten1 = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.rnn_layer_num = 2
        self.rnn_hidden_dim = 10
        self.rnn_layer = nn.GRU(kernel_num, self.rnn_hidden_dim, self.rnn_layer_num, dropout=0.5, batch_first=True,
                                bidirectional=True)
        self.rnn2output = nn.Linear(self.rnn_hidden_dim * 2, 1)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.rnn_layer]:
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, sequence, hidden=None):
        multi_time = sequence.shape[1] - self.left - self.right
        cnn_output = torch.zeros([sequence.shape[0], sequence.shape[1], 5 * self.kernel_num], device='cuda')
        sequence = torch.unsqueeze(sequence, 1)
        for i_time in range(multi_time):
            i_output = self.conv1(sequence[:, :, i_time:i_time + self.left + self.right, :])
            i_output = self.relu(i_output)
            i_output = self.bn2(self.conv2(i_output))
            i_output = self.relu(i_output)
            # i_output = self.pool2(i_output)
            i_output = self.bn3(self.conv3(i_output))
            i_output = self.bn4(self.conv4(i_output))
            # i_output = self.relu(i_output)
            # i_output = self.pool3(i_output)
            # i_output = self.relu(i_output)
            i_output = self.flatten1(i_output)
            cnn_output[:, i_time + self.left, :] = i_output
        cnn_output = self.cnn2rnn(cnn_output)
        cnn_output = self.relu(cnn_output)
        rnn_out, hidden = self.rnn_layer(cnn_output, hidden)
        output = self.rnn2output(rnn_out)
        return output


class TianModel(BaseModel):
    def __init__(self, data_path, x_fields, y_fields, weights, base_scalar):
        BaseModel.__init__(self, data_path, x_fields, y_fields, weights, base_scalar)
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3

    # def preprocess_train_data(self, x, y, weight):
    #     x = self.normalize_data(x, self._data_scalar, 'fit_transform')
    #     y['output'] = self.resample_stance_phase_kam(y['output'], weight['output'])
    #     return x, y
    #
    # def preprocess_validation_test_data(self, x, y, weight):
    #     x = self.normalize_data(x, self._data_scalar, 'transform')
    #     y['output'] = self.resample_stance_phase_kam(y['output'], weight['output'])
    #     return x, y

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
        left, right = 15, 15
        nn_model = TianCNN4(48, 1)

        if USE_GPU:
            nn_model = nn_model.cuda()
        summary(nn_model, x_train.shape[1:])
        pytorch_total_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
        logging.info('Model has {} parameters.'.format(pytorch_total_params))

        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=2e-5, weight_decay=2e-6)
        # optimizer = torch.optim.Adam(nn_model.parameters())

        batch_size = 20
        train_ds = TensorDataset(x_train, y_train, torch.from_numpy(np.zeros([x_train.shape[0]])))
        train_size = int(0.95 * len(train_ds))
        vali_from_train_size = len(train_ds) - train_size
        train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds,
                                                                             [train_size, vali_from_train_size])
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=batch_size)

        x_validation = torch.from_numpy(x_validation).float()
        y_validation = torch.from_numpy(y_validation).float()
        vali_step_lens = torch.from_numpy(self.validation_step_lens)
        vali_from_test_ds = TensorDataset(x_validation, y_validation,
                                          torch.from_numpy(np.zeros([x_validation.shape[0]])))
        num_of_step_for_peek = int(0.05 * len(x_validation))
        vali_from_test_ds, _ = torch.utils.data.dataset.random_split(vali_from_test_ds, [num_of_step_for_peek, len(
            x_validation) - num_of_step_for_peek])
        vali_from_test_dl = DataLoader(vali_from_test_ds, batch_size=batch_size)

        logging.info('\tEpoch\t\tTrain_Loss\tVali_train_Loss\tVali_test_Loss\t\tDuration\t\t')
        for epoch in range(30):
            epoch_start_time = time.time()
            for i_batch, (xb, yb, lens) in enumerate(train_dl):
                # if i_batch > 1:
                #     n = random.randint(1, 100)
                #     if n > 20:
                #         continue        # increase the speed of epoch

                if USE_GPU:
                    xb = xb.cuda()
                    yb = yb.cuda()

                optimizer.zero_grad()

                # # For RNN
                # hidden = nn_model.init_hidden(batch_size)
                # y_pred, _ = nn_model(xb, hidden, lens)

                y_pred = nn_model(xb)

                # train_loss = self.loss_fun_emphasize_peak(y_pred, yb)
                # train_loss = self.loss_fun_only_positive(y_pred, yb)
                train_loss = loss_fn(y_pred, yb)

                if epoch == 0 and i_batch == 0:
                    vali_from_train_loss = TianModel.evaluate_validation_set(nn_model, vali_from_train_dl, loss_fn,
                                                                             batch_size)
                    vali_from_test_loss = TianModel.evaluate_validation_set(nn_model, vali_from_test_dl, loss_fn,
                                                                            batch_size)
                    logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t\t".format(
                        epoch, train_loss.item(), vali_from_train_loss, vali_from_test_loss,
                        time.time() - epoch_start_time))
                train_loss.backward()
                optimizer.step()

            vali_from_train_loss = TianModel.evaluate_validation_set(nn_model, vali_from_train_dl, loss_fn, batch_size)
            vali_from_test_loss = TianModel.evaluate_validation_set(nn_model, vali_from_test_dl, loss_fn, batch_size)
            logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t\t".format(
                epoch, train_loss.item(), vali_from_train_loss.item(), vali_from_test_loss.item(),
                time.time() - epoch_start_time))
        return nn_model

    @staticmethod
    def evaluate_validation_set(nn_model, validation_dl, loss_fn, batch_size):
        validation_loss = []
        for x_validation, y_validation, lens in validation_dl:
            if USE_GPU:
                x_validation = x_validation.cuda()
                y_validation = y_validation.cuda()

            with torch.no_grad():
                # # For RNN
                # hidden = nn_model.init_hidden(x_validation.shape[0])
                # y_validation_pred, _ = nn_model(x_validation, hidden, lens)

                y_validation_pred = nn_model(x_validation)

                validation_loss.append(loss_fn(y_validation_pred, y_validation).item())
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

            test_ds = TensorDataset(x_test)
            test_dl = DataLoader(test_ds, batch_size=50)
            y_pred_list = []
            for i_batch, xb in enumerate(test_dl):
                y_pred_list.append(nn_model(xb[0]).detach().cpu())
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
    IMU_FIELDS_ACC = ['AccelX', 'AccelY', 'AccelZ']
    IMU_FIELDS_GYR = ['GyroX', 'GyroY', 'GyroZ']
    IMU_DATA_FIELDS_ACC = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_ACC]
    IMU_DATA_FIELDS_GYR = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_GYR]
    video_cols = [loc + '_' + axis + '_' + angle for loc in ['RHip', 'LHip', 'RKnee', 'LKnee']
                  for angle in ['90', '180'] for axis in ['x', 'y']]
    output_cols = ['RIGHT_KNEE_ADDUCTION_MOMENT', 'EXT_KM_Y']

    x_fields = {'main_input_acc': IMU_DATA_FIELDS_ACC,
                'main_input_gyr': IMU_DATA_FIELDS_GYR,
                # 'main_input_vid': video_cols,
                # 'aux_input': [SUBJECT_WEIGHT, SUBJECT_HEIGHT]
                }
    MAIN_TARGETS_LIST = ['RIGHT_KNEE_ADDUCTION_MOMENT']
    y_fields = {'main_output': MAIN_TARGETS_LIST}
    weights = {'main_output': [FORCE_PHASE] * len(output_cols)}
    model = TianModel(data_path, x_fields, y_fields, weights, lambda: MinMaxScaler(feature_range=(-3, 3)))
    subjects = model.get_all_subjects()
    model.preprocess_train_evaluation(subjects[:13], subjects[13:], subjects[13:])
    # model.cross_validation(subjects)
    plt.show()
