from base_kam_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from const import IMU_FIELDS, SENSOR_LIST

USE_GPU = True


class TianRNN(nn.Module):
    def __init__(self, x_dim, hidden_dim, nlayer, y_dim):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.lstm = nn.LSTM(x_dim, hidden_dim, nlayer, batch_first=True)
        self.hidden2output = nn.Linear(hidden_dim, y_dim)

    def forward(self, sequence, hidden):
        lstm_out, hidden = self.lstm(sequence, hidden)
        output = self.hidden2output(lstm_out)
        # output = self.drop(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayer, batch_size, self.hidden_dim),
                weight.new_zeros(self.nlayer, batch_size, self.hidden_dim))


class WangRNN(nn.Module):
    def __init__(self, x_dim, hidden_dim, nlayer, y_dim):
        super(WangRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayer = nlayer
        self.lstm = nn.LSTM(x_dim, hidden_dim, nlayer, batch_first=True, dropout=0.5)
        self.hidden2output = nn.Linear(hidden_dim, y_dim)

    def forward(self, sequence, hidden):
        lstm_out, hidden = self.lstm(sequence, hidden)
        output = self.hidden2output(lstm_out)
        # output = self.drop(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayer, batch_size, self.hidden_dim),
                weight.new_zeros(self.nlayer, batch_size, self.hidden_dim))


class TianModel(BaseModel):
    def __init__(self):
        data_path = os.environ.get('KAM_DATA_PATH') + '/40samples+stance_swing+padding_nan.h5'
        inertial_cols = [inertial_field + '_' + sensor for sensor in SENSOR_LIST for inertial_field in IMU_FIELDS[:6]]
        video_cols = [loc + '_' + axis + '_' + angle for loc in ['RHip', 'LHip', 'RKnee', 'LKnee']
                      for angle in ['90', '180'] for axis in ['x', 'y']]
        BaseModel.__init__(self, data_path, x_fields=inertial_cols + video_cols)
        # BaseModel.__init__(self, data_path, x_fields=video_cols)
        # BaseModel.__init__(self, data_path, x_fields=inertial_cols)

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):

        # plt.figure()
        # # plt.plot(x_train[:, :, 3].ravel())
        # plt.plot(y_train.ravel())
        # plt.show()

        N_sample, D_in, D_hidden, N_layer, D_out = x_train.shape[0], x_train.shape[2], 10, 2, y_train.shape[2]
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        nn_model = TianRNN(D_in, D_hidden, N_layer, D_out)

        if USE_GPU:
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            nn_model = nn_model.cuda()
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=2e-3, weight_decay=1e-4)

        batch_size = 200
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size)

        print('Epoch', 'Loss', '\tDuration (s)', sep='\t')
        for epoch in range(20):
            epoch_start_time = time.time()
            for i_batch, (xb, yb) in enumerate(train_dl):

                optimizer.zero_grad()
                hidden = nn_model.init_hidden(xb.shape[0])
                y_pred, _ = nn_model(xb, hidden)
                loss = loss_fn(y_pred, yb)
                if epoch == 0 and i_batch == 0:
                    print(epoch, round(loss.item(), 2), round(0.0, 2), sep='\t\t')
                loss.backward()
                optimizer.step()
                # with torch.no_grad():
                #     if epoch == 0:
                #         plt.figure()
                #         # plt.plot(w1.grad)
                #         plt.plot(yb)
                #         y_pred_np = y_pred.detach().numpy()
                #         plt.plot(y_pred_np)

            print(epoch + 1, round(loss.item(), 2), round(time.time() - epoch_start_time, 2), sep='\t\t')
        return nn_model

    @staticmethod
    def predict(nn_model, x_test):
        if USE_GPU:
            nn_model = nn_model.cpu()
        x_test = torch.from_numpy(x_test)
        hidden = nn_model.init_hidden(x_test.shape[0])
        y_pred, _ = nn_model(x_test, hidden)
        y_pred = y_pred.detach().numpy()
        return y_pred

    @staticmethod
    def customized_analysis(y_test, y_pred, metrics):
        plt.figure()
        plt.plot(y_test[:, :, :].ravel())
        plt.plot(y_pred[:, :].ravel())
        pass


if __name__ == "__main__":
    model = TianModel()
    model.param_tuning(range(3, 13), range(13, 14), range(13, 16))
    plt.show()
