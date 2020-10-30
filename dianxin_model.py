import os
import matplotlib.pyplot as plt
from keras.layers import GRU, LSTM, Activation, Dense, Masking, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback
from base_kam_model import BaseModel
from const import DATA_PATH


class DXKamModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self, os.path.join(DATA_PATH, '40samples+stance_swing.h5'))

    def train_model(self, x_train, y_train, x_validation, y_validation):
        shape = x_train.shape[1:]
        model = gru_model(shape)

        model.fit(x_train, y_train, batch_size=10000, epochs=1, verbose=1,
                  callbacks=[ErrorVisualization(x_validation, y_validation)])
        return model

    def predict(self, model, y_test):
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
