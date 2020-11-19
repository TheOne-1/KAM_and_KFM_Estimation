from base_kam_model import BaseModel
import torch
import torch.nn as nn
from customized_logger import logger as logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import time
from const import IMU_FIELDS, SENSOR_LIST, DATA_PATH, SUBJECTS, PHASE, SUBJECT_WEIGHT, SUBJECT_HEIGHT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transforms3d.euler import euler2mat
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler

USE_GPU = True
CALI_VIA_GRAVITY = False


class TianRNN(nn.Module):
    def __init__(self, x_dim, hidden_dim, nlayer, y_dim):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.lstm = nn.LSTM(x_dim, hidden_dim, nlayer, dropout=0.5, batch_first=True)
        self.hidden2output = nn.Linear(hidden_dim, y_dim)

    def forward(self, sequence, hidden):
        lstm_out, hidden = self.lstm(sequence, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=230)
        output = self.hidden2output(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayer, batch_size, self.hidden_dim),
                weight.new_zeros(self.nlayer, batch_size, self.hidden_dim))


class TianModel(BaseModel):
    def __init__(self, data_path, x_fields, y_fields, weights):
        BaseModel.__init__(self, data_path, x_fields, y_fields, weights)
        if CALI_VIA_GRAVITY:
            self.cali_via_gravity()
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3

    # """ Norm each axis """
    # def preprocess_train_data(self, x_train, y_train):
    #     self.train_step_lens = self._get_step_len(x_train)
    #     return BaseModel.preprocess_train_data(self, x_train, y_train)
    #
    # def preprocess_validation_test_data(self, x, y):
    #     self.test_step_lens = self._get_step_len(x)
    #     return BaseModel.preprocess_validation_test_data(self, x, y)

    """ Norm each type of data """
    def preprocess_train_data(self, x_train, y_train):
        def transform(input_data, col, input_name):
            original_shape = input_data.shape
            input_data = input_data.reshape([-1, input_data.shape[2]])
            self._scalars[input_name] = MinMaxScaler()
            input_data[:, col] = self._scalars[input_name].fit_transform(input_data[:, col])
            input_data.reshape(original_shape)

        self.train_step_lens = self._get_step_len(x_train)
        acc_col = [i for i, x in enumerate(self._x_fields['main_input']) if 'Accel' in x]
        gyr_col = [i for i, x in enumerate(self._x_fields['main_input']) if 'Gyr' in x]
        vid_col = [i for i, x in enumerate(self._x_fields['main_input']) if '0' in x]

        input_data = x_train['main_input']
        transform(input_data, acc_col, 'acc_col')
        transform(input_data, gyr_col, 'gyr_col')
        if len(vid_col) > 0:
            transform(input_data, vid_col, 'vid_col')

        # for input_name, input_data in x_train.items():
        #     original_shape = input_data.shape
        #     input_data = input_data.reshape([-1, input_data.shape[2]])
        #     input_data = self._scalars[input_name].fit_transform(input_data)
        #     input_data = input_data.reshape(original_shape)
        #     x_train[input_name] = input_data
        return x_train, y_train

    def preprocess_validation_test_data(self, x, y):
        def transform(input_data, col, input_name):
            original_shape = input_data.shape
            input_data = input_data.reshape([-1, input_data.shape[2]])
            input_data[:, col] = self._scalars[input_name].transform(input_data[:, col])
            input_data.reshape(original_shape)

        self.test_step_lens = self._get_step_len(x)
        acc_col = [i for i, x in enumerate(self._x_fields['main_input']) if 'Accel' in x]
        gyr_col = [i for i, x in enumerate(self._x_fields['main_input']) if 'Gyr' in x]
        vid_col = [i for i, x in enumerate(self._x_fields['main_input']) if '0' in x]

        input_data = x['main_input']
        transform(input_data, acc_col, 'acc_col')
        transform(input_data, gyr_col, 'gyr_col')
        if len(vid_col) > 0:
            transform(input_data, vid_col, 'vid_col')
        return x, y

    def cali_via_gravity(self):
        for subject in SUBJECTS:
            print(subject + ' rotated')
            transform_mat_dict = self.get_static_calibration(subject)
            for sensor in SENSOR_LIST:
                transform_mat = transform_mat_dict[sensor]
                rotation_fun = lambda data: np.matmul(transform_mat, data)
                acc_cols = ['Accel' + axis + sensor for axis in ['X_', 'Y_', 'Z_']]
                acc_col_locs = [self.data_columns.index(col) for col in acc_cols]
                self._data_all_sub[subject][:, :, acc_col_locs] = np.apply_along_axis(rotation_fun, 2, self._data_all_sub[subject][:, :, acc_col_locs])
                gyr_cols = ['Gyro' + axis + sensor for axis in ['X_', 'Y_', 'Z_']]
                gyr_col_locs = [self.data_columns.index(col) for col in gyr_cols]
                self._data_all_sub[subject][:, :, gyr_col_locs] = np.apply_along_axis(rotation_fun, 2, self._data_all_sub[subject][:, :, gyr_col_locs])

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

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        x_train, y_train = x_train['main_input'], y_train['output']
        N_step, D_in, D_hidden, N_layer, D_out = x_train.shape[0], x_train.shape[2], 10, 2, y_train.shape[2]
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        train_step_lens = torch.from_numpy(self.train_step_lens)
        nn_model = TianRNN(D_in, D_hidden, N_layer, D_out)

        if USE_GPU:
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            nn_model = nn_model.cuda()
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=2e-3, weight_decay=2e-4)

        batch_size = 50
        train_ds = TensorDataset(x_train, y_train, train_step_lens)
        train_size = int(0.8 * len(train_ds))
        validation_size = len(train_ds) - train_size
        train_ds, validation_ds = torch.utils.data.dataset.random_split(train_ds, [train_size, validation_size])
        train_dl = DataLoader(train_ds, batch_size=batch_size)
        validation_dl = DataLoader(validation_ds, batch_size=validation_size)

        logging.info('\tEpoch\t\tTrain Loss\t\tValidation Loss\t\tDuration\t\t')
        for epoch in range(10):
            epoch_start_time = time.time()
            for i_batch, (xb, yb, lens) in enumerate(train_dl):

                optimizer.zero_grad()
                hidden = nn_model.init_hidden(xb.shape[0])
                xb = pack_padded_sequence(xb, lens, batch_first=True, enforce_sorted=False)
                y_pred, _ = nn_model(xb, hidden)
                # train_loss = self.loss_fun_emphasize_peak(y_pred, yb)
                # train_loss = self.loss_fun_only_positive(y_pred, yb)
                train_loss = loss_fn(y_pred, yb)

                if epoch == 0 and i_batch == 0:
                    logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t\t{:13.2f}s\t\t\t".format(
                        epoch, train_loss.item(), 0.0, time.time() - epoch_start_time))
                    # print(epoch, round(train_loss.item(), 2), round(0.0, 2), sep='\t\t')
                train_loss.backward()
                optimizer.step()

            validation_loss = TianModel.evaluate_validation_set(nn_model, validation_dl, loss_fn, batch_size)
            # validation_loss = train_loss
            logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t\t{:13.2f}s\t\t\t".format(
                epoch, train_loss.item(), validation_loss.item(), time.time() - epoch_start_time))
        return nn_model

    @staticmethod
    def get_static_calibration(subject_name):
        data_path = DATA_PATH + '/' + subject_name + '/combined/static_side.csv'
        static_data = pd.read_csv(data_path, index_col=0)
        transform_mat_dict = {}
        for sensor in SENSOR_LIST:
            acc_cols = ['Accel' + axis + sensor for axis in ['X_', 'Y_', 'Z_']]
            acc_mean = np.mean(static_data[acc_cols], axis=0)
            roll = np.arctan2(acc_mean[1], acc_mean[2])
            pitch = np.arctan2(-acc_mean[0], np.sqrt(acc_mean[1] ** 2 + acc_mean[2] ** 2))
            # print(np.rad2deg(roll), np.rad2deg(pitch))
            transform_mat_dict[sensor] = euler2mat(roll, pitch, 0)
        return transform_mat_dict

    @staticmethod
    def evaluate_validation_set(nn_model, validation_dl, loss_fn, batch_size):
        for x_validation, y_validation, lens in validation_dl:
            hidden = nn_model.init_hidden(x_validation.shape[0])
            x_validation = pack_padded_sequence(x_validation, lens, batch_first=True, enforce_sorted=False)
            y_validation_pred, _ = nn_model(x_validation, hidden)
            validation_loss = loss_fn(y_validation_pred, y_validation) / len(y_validation) * batch_size
            return validation_loss

    def predict(self, nn_model, x_test):
        x_test = x_test['main_input']
        x_test = torch.from_numpy(x_test)
        if USE_GPU:
            x_test = x_test.cuda()
        hidden = nn_model.init_hidden(x_test.shape[0])
        x_test = pack_padded_sequence(x_test, self.test_step_lens, batch_first=True, enforce_sorted=False)
        y_pred, _ = nn_model(x_test, hidden)
        y_pred = y_pred.detach().cpu().numpy()
        return {'output': y_pred}

    @staticmethod
    def _get_step_len(data, input_cate='main_input', feature_col_num=0):
        """

        :param data: Numpy array, 3d (step, sample, feature)
        :param feature_col_num: int, feature column id for step length detection. Different id would probably return
               the same results
        :return:
        """
        data_the_feature = data[input_cate][:, :, feature_col_num]
        nan_loc = np.isnan(data_the_feature)
        data_len = np.sum(~nan_loc, axis=1)
        return data_len


if __name__ == "__main__":

    data_path = DATA_PATH + '/40samples+stance_swing+padding_zero.h5'
    inertial_cols = [inertial_field + '_' + sensor for sensor in SENSOR_LIST for inertial_field in IMU_FIELDS[:6]]
    video_cols = [loc + '_' + axis + '_' + angle for loc in ['RHip', 'LHip', 'RKnee', 'LKnee']
                  for angle in ['90', '180'] for axis in ['x', 'y']]
    output_cols = ['RIGHT_KNEE_ADDUCTION_MOMENT']

    x_fields = {'main_input': inertial_cols+video_cols, 'aux_input': [SUBJECT_WEIGHT, SUBJECT_HEIGHT]}
    y_fields = {'output': output_cols}
    weights = {'output': [PHASE]*len(output_cols)}

    model = TianModel(data_path, x_fields, y_fields, weights)
    # model.preprocess_train_evaluation(range(10), [], range(10, 13))
    model.cross_validation(range(len(SUBJECTS)))
    plt.show()
