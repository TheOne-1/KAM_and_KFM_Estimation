import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from customized_logger import logger as logging
import tensorflow as tf
import numpy as np
from keras import Model, Input
from keras.layers import GRU, LSTM, Activation, Dense, Masking, Conv1D, Bidirectional, GaussianNoise, MaxPooling1D
from keras.layers import UpSampling1D
import keras.backend as K
import keras.losses as Kloss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from base_kam_model import BaseModel
from const import DATA_PATH, SENSOR_LIST, VIDEO_LIST, TARGETS_LIST

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
IMU_DATA_FIELDS = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS]
VIDEO_DATA_FIELDS = [VIDEO + "_" + position + "_" + angle for VIDEO in VIDEO_LIST
                     for position in ["x", "y"] for angle in ["90", "180"]]


class DXKamModelWithAutoencoder(BaseModel):
    def __init__(self, data_file, x_fields, y_fields):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), x_fields, y_fields, StandardScaler)
        # TODO: feed IMU data fields and Video data fields into two separated network at the beginning.
        self.imu_autoencoder = None

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        input_shape = x_train.shape[1:]
        model = rnn_model(input_shape, LSTM)
        validation_data = None
        callbacks = []
        if x_validation is not None:
            validation_data = (x_validation, y_validation)
            # callbacks.append(ErrorVisualization(x_validation, y_validation))
            callbacks.append(ReduceLROnPlateau('val_loss', factor=0.1, patience=5))
        model.fit(x_train, y_train, validation_data=validation_data, shuffle=True, batch_size=30,
                       epochs=20, verbose=1, callbacks=callbacks)
        return model

    def preprocess_train_data(self, x_train, y_train):
        x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
        input_shape = x_train.shape[1:]
        self.imu_autoencoder = autoencoder(input_shape)
        self.imu_autoencoder.fit(x_train, x_train, shuffle=True, batch_size=20, epochs=30, verbose=1, callbacks=[PlotVisualization(x_train, x_train)])
        x_train = np.multiply(self.imu_autoencoder.predict(x_train), (x_train != 0.).all(axis=2)[:, :, None])
        return x_train, y_train

    def preprocess_validation_test_data(self, x_train, y_train):
        x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
        x_train = np.multiply(self.imu_autoencoder.predict(x_train), (x_train != 0.).all(axis=2)[:, :, None])
        return x_train, y_train

    @staticmethod
    def predict(model, x_test):
        return model.predict(x_test, verbose=1)


class DXKamModel(BaseModel):
    def __init__(self, data_file, x_fields, y_fields):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), x_fields, y_fields, StandardScaler)
        # TODO: feed IMU data fields and Video data fields into two separated network at the beginning.

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        input_shape = x_train.shape[1:]
        model = rnn_model(input_shape, LSTM)
        validation_data = None
        callbacks = []
        if x_validation is not None:
            validation_data = (x_validation, y_validation)
            callbacks.append(ErrorVisualization(x_validation, y_validation))
            callbacks.append(ReduceLROnPlateau('val_loss', factor=0.1, patience=5))
        model.fit(x_train, y_train, validation_data=validation_data, shuffle=True, batch_size=30,
                       epochs=20, verbose=1, callbacks=callbacks)
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
        BaseModel.representative_profile_curves(y_true, y_pred, "Validation results")

    def on_epoch_begin(self, epoch, logs=None):
        y_true = self.Y_test
        y_pred = self.model.predict(self.X_test)
        scores = BaseModel.get_all_scores(y_true, y_pred)
        logging.info("initial scores: {}".format(scores))
        time.sleep(0.1)  # Make sure the logging is printed first

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.Y_test
        y_pred = self.model.predict(self.X_test)
        BaseModel.representative_profile_curves(y_true, y_pred, "Validation results")


class PlotVisualization(Callback):
    def __init__(self, x_test, y_test):
        Callback.__init__(self)
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.y_test
        y_pred = self.model.predict(self.x_test)
        BaseModel.representative_profile_curves(y_true, y_pred, "Validation results")


# baseline model
def rnn_model(input_shape, rnn_layer):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(GaussianNoise(0.01))
    model.add(Bidirectional(rnn_layer(96, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(rnn_layer(30, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(rnn_layer(15, dropout=0.2, return_sequences=True)))
    model.add(Dense(20, use_bias=True, activation='tanh'))
    model.add(Dense(1, use_bias=True))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


def autoencoder(input_shape):
    input_ts = Input(shape=input_shape)
    x = Conv1D(32, 5, activation="relu", padding="same")(input_ts)
    x = MaxPooling1D(2, strides=2, padding="same")(x)
    x = Conv1D(16, 5, activation="relu", padding="same")(x)
    encoded = MaxPooling1D(2, strides=2, padding="same")(x)
    encoder = Model(input_ts, encoded)

    x = Conv1D(16, 5, activation="relu", padding="same")(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 5, activation='relu', padding="same")(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(input_shape[-1], 1, activation='tanh', padding='same')(x)

    convolutional_autoencoder = Model(input_ts, decoded)

    convolutional_autoencoder.summary()

    optimizer = "adam"
    def custom_loss(y_true, y_pred):
        temp = K.cast(y_true != 0, 'float32')
        temp = K.flatten(temp)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        y_pred_f = y_pred_f * temp
        return Kloss.mean_absolute_error(y_true_f, y_pred_f)
    convolutional_autoencoder.compile(optimizer=optimizer, loss=custom_loss)
    return convolutional_autoencoder


if __name__ == "__main__":
    dx_model = DXKamModel('40samples+stance_swing+kick_out_trunksway.h5', IMU_DATA_FIELDS,
                          TARGETS_LIST)
    # dx_model.param_tuning(range(11), range(11, 13), range(11, 13))
    dx_model.cross_validation(range(13))
    # dx_model = DXKamModelWithAutoencoder('40samples+stance_swing+kick_out_trunksway.h5', IMU_DATA_FIELDS, TARGETS_LIST)
    # dx_model.param_tuning(range(11), range(11, 13), range(11, 13))
