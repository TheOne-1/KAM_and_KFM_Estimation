from base_kam_model import BaseModel
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class TianModel(BaseModel):
    @staticmethod
    def preprocessing(train_data_list, validate_data_list, test_data_list):
        train_data_list = [np.nan_to_num(data) for data in train_data_list]
        validate_data_list = [np.nan_to_num(data) for data in validate_data_list]
        test_data_list = [np.nan_to_num(data) for data in test_data_list]

        return train_data_list, validate_data_list, test_data_list

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):
        x_train = x_train.reshape(-1, x_train.shape[1] * x_train.shape[2])
        y_train = y_train.reshape(-1, y_train.shape[1] * y_train.shape[2])
        # x_train = x_train.reshape([-1, x_train.shape[2]])
        # y_train = y_train.reshape([-1, 1])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train)

        N, D_in, H, D_out = x_train.shape[0], x_train.shape[1], 10, y_train.shape[1]
        learning_rate = 1e-9

        w1 = np.random.randn(D_in, H)
        w2 = np.random.randn(H, D_out) * 1e-3
        for epoch in range(5):
            h = x_train.dot(w1)
            h_relu = np.maximum(h, 0)
            y_pred = h_relu.dot(w2)
            loss = np.square(y_pred - y_train).sum()
            print(epoch, loss)

            grad_y_pred = 2 * (y_pred - y_train)
            grad_w2 = h_relu.T.dot(grad_y_pred)
            grad_h_relu = grad_y_pred.dot(w2.T)
            grad_h = grad_h_relu.copy()
            grad_h[h < 0] = 0
            grad_w1 = x_train.T.dot(grad_h)

            plt.figure()
            plt.plot(y_train[1000:1003].ravel())
            plt.plot(y_pred[1000:1003].ravel())

            # Update weights
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
        model = [w1, w2, scalar]
        return model

    @staticmethod
    def predict(model, x_test):
        w1, w2, scalar = model

        x_test = x_test.reshape(-1, x_test.shape[1] * x_test.shape[2])
        # x_test = x_test.reshape([-1, x_test.shape[2]])
        x_test = scalar.transform(x_test)

        h = x_test.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)
        # y_pred = y_pred.reshape([-1, 160, 1])

        return y_pred



if __name__ == "__main__":
    model = TianModel()
    model.param_tuning(range(12), range(12, 13), range(13, 16))
    plt.show()

