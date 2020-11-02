from base_kam_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os


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


class TianModel(BaseModel):
    def __init__(self):
        data_path = os.environ.get('KAM_DATA_PATH') + '/40samples+stance_swing.h5'
        BaseModel.__init__(self, data_path)

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):
        # x_train, y_train = np.nan_to_num(x_train), np.nan_to_num(y_train)
        # x_validation, y_validation = np.nan_to_num(x_validation), np.nan_to_num(y_validation)
        N, D_in, D_hidden, D_out = x_train.shape[0], x_train.shape[2], 2, y_train.shape[2]
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        nn_model = TianRNN(D_in, D_hidden, 1, D_out)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 5e-2
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

        batch_size = 1000
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size)

        for epoch in range(2):
            for i_batch, (xb, yb) in enumerate(train_dl):
                optimizer.zero_grad()
                hidden = nn_model.init_hidden(xb.shape[0])
                y_pred, _ = nn_model(xb, hidden)
                loss = loss_fn(y_pred, yb)
                if i_batch == 0:
                    print(epoch, loss.item())
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
        # x_test = np.nan_to_num(x_test)
        x_test = torch.from_numpy(x_test)
        hidden = nn_model.init_hidden(x_test.shape[0])
        y_pred, _ = nn_model(x_test, hidden)
        y_pred = y_pred.detach().numpy()
        return y_pred

    @staticmethod
    def representative_profile_curves(y_test, y_pred, metrics):
        # y_test = np.nan_to_num(y_test)
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

