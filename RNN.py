from keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM, Activation, Dense, Masking, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import h5py
import numpy as np

from generate_step_data import get_step_data

subject_data_dict = get_step_data('/media/dianxin/Software/whole_data_160.h5')

subject_data = list(subject_data_dict.values())
train_data = np.concatenate(subject_data[1:], axis=0)
test_data = np.concatenate(subject_data[:1], axis=0)
np.random.shuffle(train_data)
np.random.shuffle(test_data)
[X_train, Y_train, X_test, Y_test] = [train_data[:, :, 0:-1], train_data[:, :, -1:],
                                      test_data[:, :, 0:-1], test_data[:, :, -1:]]
SHAPE = X_train.shape[1:]


# pad the sequences with zeros
# padding parameter is set to 'post' => 0's are appended to end of sequences
# X_train = pad_sequences(X_train, maxlen = maxlen, padding = 'post')
# X_test = pad_sequences(X_test, maxlen = maxlen, padding = 'post')

class ErrorVisualization(Callback):
    def __init__(self, X_test, Y_test):
        Callback.__init__(self)
        self.X_test = X_test
        Y_test_mean = Y_test.mean(axis=0).reshape((-1))
        Y_test_std = Y_test.std(axis=0).reshape((-1))
        self.axis_x = range(Y_test_mean.shape[0])
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.plot(self.axis_x, Y_test_mean, 'g-', label='Real_Value')
        self.ax.fill_between(self.axis_x, Y_test_mean - Y_test_std, Y_test_mean + Y_test_std, facecolor='green',
                             alpha=0.2)

    def on_train_begin(self, logs={}):
        Y_predict = self.model.predict(X_test)
        Y_predict_mean = Y_predict.mean(axis=0).reshape((-1))
        Y_predict_std = Y_predict.std(axis=0).reshape((-1))
        self.y_predict_mean_line = self.ax.plot(self.axis_x, Y_predict_mean, 'y-', label='Predict_Value')
        self.y_predict_std_area = self.ax.fill_between(self.axis_x, Y_predict_mean - Y_predict_std,
                                                       Y_predict_mean + Y_predict_std, facecolor='yellow', alpha=0.2)
        self.fig.canvas.draw()

    def on_epoch_end(self, epoch, logs={}):
        Y_predict = self.model.predict(X_test)
        Y_predict_mean = Y_predict.mean(axis=0).reshape((-1))
        Y_predict_std = Y_predict.std(axis=0).reshape((-1))

        self.y_predict_std_area.remove()
        self.y_predict_mean_line[0].set_ydata(Y_predict_mean)
        self.y_predict_std_area = self.ax.fill_between(self.axis_x, Y_predict_mean - Y_predict_std,
                                                       Y_predict_mean + Y_predict_std, facecolor='yellow', alpha=0.2)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()


def gru_model():
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=SHAPE))
    model.add(GRU(20, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(30, dropout=0.2, return_sequences=True))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


model = gru_model()

model.fit(X_train, Y_train, batch_size=10, epochs=20, verbose=1, callbacks=[ErrorVisualization(X_test, Y_test)])

scores = model.evaluate(X_test, Y_test, verbose=1)