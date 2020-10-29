import os
import numpy as np
import ipykernel  # make the terminal output within one line
import matplotlib.pyplot as plt
from keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM, Activation, Dense, Masking, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback
from base_kam_model import BaseModel
from config import DATA_PATH


class DXKamModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self, os.path.join(DATA_PATH, '40samples+stance_swing.h5'))

    @staticmethod
    def preprocessing(train_data, validation_data, test_data):
        # # normalize video data
        # for subject, data_collections in subject_data_dict.items():
        #     for data in data_collections:
        #         for angle in ["90", "180"]:
        #             for position in ["x", "y"]:
        #                 angle_specific_video_data_fields = [VIDEO + "_" + position + "_" + angle for VIDEO in
        #                                                     VIDEO_LIST]
        #                 data.loc[:, angle_specific_video_data_fields] -= \
        #                     data.loc[:, "MidHip_" + position + "_" + angle].mean(axis=0)
        #                 data.loc[:, angle_specific_video_data_fields] /= 1920
        #                 data.loc[:, angle_specific_video_data_fields] += 0.5
        #
        # for subject, data_collections in subject_data_dict.items():
        #     for data in data_collections:
        #         data.loc[:, IMU_DATA_FIELDS] -= data.loc[:, IMU_DATA_FIELDS].mean(axis=0)
        #         data.loc[:, IMU_DATA_FIELDS] /= data.loc[:, IMU_DATA_FIELDS].std(axis=0)

        for data in [train_data, validation_data, test_data]:
            for subject_data in data:
                subject_data[np.isnan(subject_data)] = 0
        return train_data, validation_data, test_data

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):
        shape = x_train.shape[1:]
        model = gru_model(shape)

        model.fit(x_train, y_train, batch_size=10000, epochs=1, verbose=1,
                  callbacks=[ErrorVisualization(x_validation, y_validation)])
        return model

    @staticmethod
    def predict(model, y_test):
        return model.predict(y_test, verbose=1)


class ErrorVisualization(Callback):
    def __init__(self, x_test, y_test):
        Callback.__init__(self)
        self.X_test = x_test
        self.Y_test = y_test

    def plot_std_result(self):
        y_predict = self.model.predict(self.X_test)
        y_predict_mean = y_predict.mean(axis=0).reshape((-1))
        y_predict_std = y_predict.std(axis=0).reshape((-1))

        y_test_mean = self.Y_test.mean(axis=0).reshape((-1))
        y_test_std = self.Y_test.std(axis=0).reshape((-1))
        axis_x = range(y_test_mean.shape[0])
        plt.plot(axis_x, y_test_mean, 'g-', label='Real_Value')
        plt.fill_between(axis_x, y_test_mean - y_test_std, y_test_mean + y_test_std, facecolor='green',
                         alpha=0.2)
        plt.plot(axis_x, y_predict_mean, 'y-', label='Predict_Value')
        plt.fill_between(axis_x, y_predict_mean - y_predict_std, y_predict_mean + y_predict_std,
                         facecolor='yellow', alpha=0.2)
        plt.show(block=False)

    def on_train_begin(self, logs=None):
        self.plot_std_result()

    def on_epoch_end(self, epoch, logs=None):
        self.plot_std_result()


def gru_model(input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(GRU(20, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(30, dropout=0.2, return_sequences=True))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


if __name__ == "__main__":
    dx_model = DXKamModel()
    dx_model.param_tuning(range(0, 14), range(14, 15), range(15, 16))
