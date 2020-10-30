from base_kam_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os


class OneLayerAnnModel(BaseModel):
    @staticmethod
    def preprocess_train_data(train_data_list, validate_data_list, test_data_list):
        # train_data_list = [np.nan_to_num(data[:, :60, :]) for data in train_data_list]
        # validate_data_list = [np.nan_to_num(data[:, :60, :]) for data in validate_data_list]
        # test_data_list = [np.nan_to_num(data[:, :60, :]) for data in test_data_list]
        train_data_list = [np.nan_to_num(data[:, 30:80, :]) for data in train_data_list]
        validate_data_list = [np.nan_to_num(data[:, 30:80, :]) for data in validate_data_list]
        test_data_list = [np.nan_to_num(data[:, 30:80, :]) for data in test_data_list]

        return train_data_list, validate_data_list, test_data_list

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):
        x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
        y_train = y_train.reshape(-1, y_train.shape[1] * y_train.shape[2])
        # x_train = x_train.reshape([-1, x_train.shape[2]])
        # y_train = y_train.reshape([-1, 1])

        x_scalar = MinMaxScaler()
        x_train = x_scalar.fit_transform(x_train)
        # y_scalar = MinMaxScaler(feature_range=(0, 1))
        # y_train = y_scalar.fit_transform(y_train)
        y_train = y_train

        N, D_in, H, D_out = x_train.shape[0], x_train.shape[1], 10, y_train.shape[1]
        learning_rate = 1e-1
        dtype = torch.float

        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        w1 = torch.randn(D_in, H, requires_grad=True, dtype=dtype)
        w2 = torch.randn(H, D_out, requires_grad=True, dtype=dtype)
        for epoch in range(50):
            y_pred = x_train.mm(w1).clamp(min=0).mm(w2 * 1e-5)      # NO idea why w2 needs * 1e-2
            loss = (y_pred - y_train).pow(2).sum()
            print(epoch, loss.item())
            loss.backward()

            # Update weights
            with torch.no_grad():
                if epoch % 10 == 0:
                    plt.figure()
                    # plt.plot(w1.grad)
                    plt.plot(y_train[2000, :])
                    y_pred_np = y_pred.detach().numpy()
                    plt.plot(y_pred_np[2000, :])

                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad
                w1.grad.zero_()
                w2.grad.zero_()
        model = [w1, w2, x_scalar, None]
        # model = [w1, w2, x_scalar, y_scalar]
        return model

    @staticmethod
    def predict(model, x_test):
        w1, w2, x_scalar, y_scalar = model

        x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
        x_test = x_scalar.transform(x_test)

        x_test = torch.from_numpy(x_test)

        y_pred = x_test.mm(w1).clamp(min=0).mm(w2 * 1e-5)
        y_pred = y_pred.detach().numpy()
        return y_pred

    @staticmethod
    def representative_profile_curves(y_test, y_pred, metrics):
        plt.figure()
        # plt.plot(w1.grad)
        plt.plot(y_test[:, :, :].ravel())
        plt.plot(y_pred[:, :].ravel())
        pass


class TianRNN(nn.Module):
    def __init__(self, x_dim, hidden_dim, y_dim):
        super(TianRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(x_dim, hidden_dim, 2)
        self.hidden2output = nn.Linear(hidden_dim, y_dim)

    def forward(self, sequence, hidden):
        lstm_out, hidden = self.lstm(sequence, hidden)
        output = self.hidden2output(lstm_out.view(len(sequence), -1))
        # output = self.drop(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))


class TianModel(BaseModel):
    def __init__(self):
        data_path = os.environ.get('KAM_DATA_PATH') + '/40samples+stance_swing.h5'
        BaseModel.__init__(self, data_path)

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):


        N, D_in, H, D_out = x_train.shape[0], x_train.shape[1], 100, y_train.shape[1]
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=200)

        nn_model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H,  bias=False),
            torch.nn.Sigmoid(),
            torch.nn.Linear(H, D_out, bias=False),
        )
        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 5e-2
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

        for epoch in range(5):
            for xb, yb in train_dl:
                y_pred = nn_model(xb)
                loss = loss_fn(y_pred, yb)

                print(epoch, loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # with torch.no_grad():
                #     if epoch == 0:
                #         plt.figure()
                #         # plt.plot(w1.grad)
                #         plt.plot(yb)
                #         y_pred_np = y_pred.detach().numpy()
                #         plt.plot(y_pred_np)
        return nn_model

    @staticmethod
    def predict(nn_model, x_test):
        # x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
        # x_test = x_scalar.transform(x_test)

        x_test = torch.from_numpy(x_test)

        y_pred = nn_model(x_test)
        y_pred = y_pred.detach().numpy()
        return y_pred

    @staticmethod
    def representative_profile_curves(y_test, y_pred, metrics):
        plt.figure()
        # plt.plot(w1.grad)
        plt.plot(y_test[:, :, :].ravel())
        plt.plot(y_pred[:, :].ravel())
        pass


if __name__ == "__main__":
    model = TianModel()
    # plt.ion()
    model.param_tuning(range(12), range(12, 13), range(13, 16))
    plt.show()

