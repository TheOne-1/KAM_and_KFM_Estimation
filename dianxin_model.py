import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from customized_logger import logger as logging
import tensorflow as tf
from keras.layers import GRU, LSTM, Activation, Dense, Masking, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback
from base_kam_model import BaseModel
from const import DATA_PATH, SENSOR_LIST, VIDEO_LIST, TARGETS_LIST

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
IMU_DATA_FIELDS = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS]
VIDEO_DATA_FIELDS = [VIDEO + "_" + position + "_" + angle for VIDEO in VIDEO_LIST
                     for position in ["x", "y"] for angle in ["90", "180"]]


class DXKamModel(BaseModel):
    def __init__(self, model_callback, data_file, x_fields, y_fields):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), x_fields, y_fields)
        self.model_callback = model_callback

    def train_model(self, x_train, y_train, x_validation, y_validation):
        # TODO: feed IMU data fields and Video data fields into two separated network at the beginning.
        shape = x_train.shape[1:]
        model = self.model_callback(shape)

        # You might see this WARNING:
        # Allocation of 1823325480 exceeds 10% of free system memory.
        # This is because you batch size is so large. Try to use 20 or similar.
        model.fit(x_train, y_train, validation_data=(x_validation, y_validation), shuffle=True, batch_size=50,
                  epochs=30, verbose=1,
                  callbacks=[ErrorVisualization(x_validation, y_validation)])
        return model

    @staticmethod
    def predict(model, x_test):
        return model.predict(x_test, verbose=1)


class ErrorVisualization(Callback):
    def __init__(self, x_test, y_test):
        Callback.__init__(self)
        self.X_test = x_test
        self.Y_test = y_test

    def on_train_begin(self, logs=None):
        y_true = self.Y_test
        y_pred = self.model.predict(self.X_test)
        BaseModel.representative_profile_curves(y_true, y_pred, {})

    def on_epoch_begin(self, epoch, logs=None):
        y_true = self.Y_test
        y_pred = self.model.predict(self.X_test)
        scores = BaseModel.get_all_scores(y_true, y_pred)
        logging.info("initial scores: {}".format(scores))

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.Y_test
        y_pred = self.model.predict(self.X_test)
        BaseModel.representative_profile_curves(y_true, y_pred, {})


def gru_model(input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(GRU(20, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(30, dropout=0.2, return_sequences=True))
    model.add(Dense(5, use_bias=True, activation='tanh'))
    model.add(Dense(1, use_bias=True))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


def dense_model(input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(Dense(5, use_bias=True, activation='tanh'))
    model.add(Dense(1, use_bias=True))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


if __name__ == "__main__":
    GOOD_SUBJECTS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    dx_model = DXKamModel(gru_model, '40samples+stance_swing+padding_nan.h5', IMU_DATA_FIELDS + VIDEO_DATA_FIELDS,
                          TARGETS_LIST)
    dx_model.param_tuning(GOOD_SUBJECTS[0:-1], GOOD_SUBJECTS[-1:], GOOD_SUBJECTS[-1:])
