import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from customized_logger import logger as logging
import tensorflow as tf
import numpy as np
from random import shuffle
from keras import Model, Input
from keras.layers import GRU, LSTM, Dense, Masking, Conv1D, Bidirectional, GaussianNoise, MaxPooling1D
from keras.layers import UpSampling1D, concatenate, RepeatVector
import keras.backend as K
import keras.losses as Kloss
from sklearn.preprocessing import StandardScaler  # MinMaxScaler,
from keras.callbacks import Callback, ReduceLROnPlateau
from base_kam_model import BaseModel
from const import DATA_PATH, SENSOR_LIST, VIDEO_LIST, TARGETS_LIST, SUBJECT_WEIGHT, SUBJECT_HEIGHT, PHASE

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
        self.imu_autoencoder = None

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        time_length = x_train.shape[2]
        model = rnn_model(time_length, LSTM)
        validation_data = None
        callbacks = []
        if x_validation is not None:
            validation_data = (
                {'main_input': x_validation[:, :, :48], 'aux_input': x_validation[:, 0, -2:]}, y_validation)
            callbacks.append(ErrorVisualization(self, x_validation, y_validation))
            callbacks.append(ReduceLROnPlateau('val_loss', factor=0.1, patience=5))
        model.fit(x={'main_input': x_train[:, :, :48], 'aux_input': x_train[:, 0, -2:]}, y=y_train,
                  validation_data=validation_data, shuffle=True, batch_size=30,
                  epochs=30, verbose=1, callbacks=callbacks)
        return model

    def preprocess_train_data(self, x_train, y_train):
        x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
        input_shape = x_train.shape[1:]
        self.imu_autoencoder = autoencoder(input_shape)
        self.imu_autoencoder.fit(x_train, x_train, shuffle=True, batch_size=20, epochs=30, verbose=1)
        x_train = np.multiply(self.imu_autoencoder.predict(x_train), (x_train != 0.).all(axis=2)[:, :, None])
        return x_train, y_train

    def preprocess_validation_test_data(self, x_train, y_train):
        x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
        x_train = np.multiply(self.imu_autoencoder.predict(x_train), (x_train != 0.).all(axis=2)[:, :, None])
        return x_train, y_train

    def _depart_input_and_output(self, data):
        self.y_phase_data = data[:, :, [PHASE]]
        return BaseModel._depart_input_and_output(self, data)

    @staticmethod
    def predict(model, x_test):
        return model.predict({'main_input': x_test[:, :, :48], 'aux_input': x_test[:, 0, -2:]}, verbose=1)


class DXKamModel(BaseModel):
    def __init__(self, data_file, x_fields, y_fields):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), x_fields, y_fields, StandardScaler)

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        time_length = x_train.shape[1]
        model = rnn_model(time_length, GRU)
        validation_data = None
        callbacks = []
        if x_validation is not None:
            validation_data = (
                {'main_input': x_validation[:, :, :48], 'aux_input': x_validation[:, 0, -2:]}, y_validation)
            callbacks.append(ErrorVisualization(self, x_validation, y_validation))
            # callbacks.append(ReduceLROnPlateau('val_loss', factor=0.1, patience=5))
        model.fit(x={'main_input': x_train[:, :, :48], 'aux_input': x_train[:, 0, -2:]}, y=y_train,
                  validation_data=validation_data, shuffle=True, batch_size=30,
                  epochs=30, verbose=1, callbacks=callbacks)
        return model

    def _depart_input_and_output(self, data):
        self.y_phase_data = (data[:, :, self.data_columns.index(PHASE)] == 1.).astype('float32')
        return BaseModel._depart_input_and_output(self, data)

    def get_all_scores(self, y_true, y_pred, weight=None):
        return BaseModel.get_all_scores(y_true, y_pred, self.y_phase_data)

    @staticmethod
    def predict(model, x_test):
        return model.predict({'main_input': x_test[:, :, :48], 'aux_input': x_test[:, 0, -2:]}, verbose=1)


class ErrorVisualization(Callback):
    def __init__(self, model, x_test, y_test):
        Callback.__init__(self)
        self.dx_model = model
        self.X_test = x_test
        self.Y_test = y_test

    def on_epoch_begin(self, epoch, logs=None):
        y_true = self.Y_test
        y_pred = self.model.predict({'main_input': self.X_test[:, :, :48], 'aux_input': self.X_test[:, 0, -2:]})
        scores = self.dx_model.get_all_scores(y_true, y_pred)
        self.dx_model.representative_profile_curves(y_true[:, :, 0], y_pred[:, :, 0], "Validation results", scores[0]['r2'])
        time.sleep(0.1)  # Make sure the logging is printed first
        logging.info("initial scores: {}".format([{key: np.mean(value) for key, value in score.items()} for score in scores]))


# baseline model
def rnn_model(time_length, rnn_layer):
    dynamic_input = Input(shape=(time_length, len(IMU_DATA_FIELDS)), name='main_input')
    x = Masking(mask_value=0.)(dynamic_input)
    x = GaussianNoise(0.02)(x)
    x = Bidirectional(rnn_layer(96, dropout=0, return_sequences=True))(x)
    x = Bidirectional(rnn_layer(30, dropout=0, return_sequences=True))(x)
    x = Bidirectional(rnn_layer(15, dropout=0, return_sequences=True))(x)
    static_input = Input(shape=(2,), name='aux_input')
    x_1 = RepeatVector(time_length)(static_input)
    x = concatenate([x, x_1])
    x = Dense(15, use_bias=True, activation='tanh')(x)
    output = Dense(len(TARGETS_LIST), use_bias=True)(x)
    model = Model(inputs=[dynamic_input, static_input], outputs=[output])

    def custom_loss(y_true, y_pred):
        temp = K.cast(y_true >= 0, 'float32')
        return K.sum(K.abs(y_true - y_pred) * temp * y_true)

    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    return model


def autoencoder(input_shape):
    input_ts = Input(shape=input_shape)
    x = Conv1D(32, 5, activation="relu", padding="same")(input_ts)
    x = MaxPooling1D(2, strides=2, padding="same")(x)
    x = Conv1D(16, 5, activation="relu", padding="same")(x)
    encoded = MaxPooling1D(2, strides=2, padding="same")(x)
    # encoder = Model(input_ts, encoded)

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
    dx_model = DXKamModel('40samples+stance_swing+padding_nan.h5', IMU_DATA_FIELDS + [SUBJECT_WEIGHT, SUBJECT_HEIGHT],
                          TARGETS_LIST)
    # dx_model.preprocess_train_evaluation(range(11), range(11, 13), range(11, 13))
    subject_list = list(range(13))
    shuffle(subject_list)
    dx_model.cross_validation(subject_list)
    # dx_model = DXKamModelWithAutoencoder('40samples+stance_swing+kick_out_trunksway.h5', IMU_DATA_FIELDS, TARGETS_LIST)
    # dx_model.cross_validation(range(13))
