import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import random
from customized_logger import logger as logging
import tensorflow as tf
from keras import Model, Input
from keras.layers import GRU, LSTM, Activation, Dense, Masking, Conv1D, Bidirectional, GaussianNoise
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.optimizers import Adam
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


class DXKamModel(BaseModel):
    def __init__(self, data_file, x_fields, y_fields):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), x_fields, y_fields)
        # TODO: feed IMU data fields and Video data fields into two separated network at the beginning.
        self.model_callback = rnn_model
        self.model = None

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        input_shape = x_train.shape[1:]
        self.model = self.model_callback(input_shape, LSTM)
        validation_data = None if x_validation is None else (x_validation, y_validation)
        self.model.fit(x_train, y_train, validation_data=validation_data, shuffle=True, batch_size=10,
                       epochs=50, verbose=1, callbacks=[ErrorVisualization(x_validation, y_validation),
                                                        ModelCheckpoint("val_best.h5", monitor='val_loss', verbose=1,
                                                                        save_best_only=True, mode='max'),
                                                        ReduceLROnPlateau('val_loss', factor=0.1, patience=5)])
        return self.model

    @staticmethod
    def predict(model, x_test):
        model.load_weights("val_best.h5")
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
        time.sleep(0.1)  # Make sure the logging is printed first

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.Y_test
        y_pred = self.model.predict(self.X_test)
        BaseModel.representative_profile_curves(y_true, y_pred, {})


# baseline model
def rnn_model(input_shape, rnn_layer):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    model.add(GaussianNoise(0.01))
    model.add(Bidirectional(rnn_layer(96, dropout=0.2, recurrent_dropout=0., return_sequences=True)))
    model.add(Bidirectional(rnn_layer(30, dropout=0.2, recurrent_dropout=0., return_sequences=True)))
    model.add(Bidirectional(rnn_layer(15, dropout=0.2, recurrent_dropout=0., return_sequences=True)))
    model.add(Dense(20, use_bias=True, activation='tanh'))
    model.add(Dense(1, use_bias=True))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


class DXAutoEncoder:
    def __init__(self, input_shape):
        encoding_dim = 32
        input_layer = Input(shape=input_shape)
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(input_shape, activation='sigmoid')(encoded)
        # This model maps an input to its reconstruction
        self.autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        decoded_input = Input(shape=(encoding_dim,))
        # Retrieve the last layer of the autoencoder model
        decoder_layer = decoded
        # Create the decoder model
        self.decoder = Model(decoded_input, decoder_layer(decoded_input))

    def encoder(self):
        return self.encoder

    def decoder(self):
        return self.decoder

    def model(self):
        return self.autoencoder


class DXKamModelWithAutoencoder(BaseModel):
    def __init__(self, data_file, x_fields, y_fields):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), x_fields, y_fields)
        # TODO: feed IMU data fields and Video data fields into two separated network at the beginning.
        shape = len(x_fields)
        self.model = rnn_model(shape, LSTM)
        self.imu_auto_encoder = DXAutoEncoder()

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        self.model.fit(x_train, y_train, validation_data=(x_validation, y_validation), shuffle=True, batch_size=20,
                       epochs=30, verbose=1,
                       callbacks=[ErrorVisualization(x_validation, y_validation)])
        return self.model

    def preprocess_train_data(self, x_train, y_train):
        x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
        self.imu_auto_encoder.fit(x_train, x_train)
        return x_train, y_train

    def preprocess_validation_test_data(self, x_train, y_train):
        x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
        return x_train, y_train

    @staticmethod
    def predict(model, x_test):
        return model.predict(x_test, verbose=1)


if __name__ == "__main__":
    GOOD_SUBJECTS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    dx_model = DXKamModel('40samples+stance_swing+padding_nan.h5', IMU_DATA_FIELDS,
                          TARGETS_LIST)
    dx_model.param_tuning(GOOD_SUBJECTS[0:-1], GOOD_SUBJECTS[-1:], GOOD_SUBJECTS[-1:])
