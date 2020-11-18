import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from random import shuffle
from keras import Model, Input
from keras.layers import GRU, Dense, Masking, Conv1D, Bidirectional, GaussianNoise, MaxPooling1D
from keras.layers import UpSampling1D, concatenate
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


class DXKamModel(BaseModel):
    def __init__(self, data_file, input_fields, output_fields, weights):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), input_fields, output_fields, weights, StandardScaler)

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None):
        model = rnn_model(x_train, y_train, GRU)
        validation_data = None
        callbacks = []
        if x_validation is not None:
            validation_data = (x_validation, y_validation)
            # callbacks.append(ErrorVisualization(self, x_validation, y_validation))
            callbacks.append(ReduceLROnPlateau('val_loss', factor=0.1, patience=5))
        model.fit(x=x_train, y=y_train, validation_data=validation_data, shuffle=True, batch_size=20,
                  epochs=30, verbose=1, callbacks=callbacks)
        return model

    @staticmethod
    def predict(model, x_test):
        prediction_list = model.predict(x_test, verbose=1)
        prediction_list = prediction_list if isinstance(prediction_list, list) else [prediction_list]
        prediction_dict = {name: pred for name, pred in zip(model.output_names, prediction_list)}
        return prediction_dict

    # def preprocess_train_data(self, x_train, y_train):
    #     x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
    #     input_shape = x_train.shape[1:]
    #     self.imu_autoencoder = autoencoder(input_shape)
    #     self.imu_autoencoder.fit(x_train, x_train, shuffle=True, batch_size=20, epochs=30, verbose=1)
    #     x_train = np.multiply(self.imu_autoencoder.predict(x_train), (x_train != 0.).all(axis=2)[:, :, None])
    #     return x_train, y_train
    #
    # def preprocess_validation_test_data(self, x_train, y_train):
    #     x_train, y_train = BaseModel.preprocess_train_data(self, x_train, y_train)
    #     x_train = np.multiply(self.imu_autoencoder.predict(x_train), (x_train != 0.).all(axis=2)[:, :, None])
    #     return x_train, y_train


class ErrorVisualization(Callback):
    def __init__(self, model, x_true, y_true):
        Callback.__init__(self)
        self.dx_model = model
        self.x_true = x_true
        self.y_true = y_true

    def on_epoch_begin(self, epoch, logs=None):
        y_pred = self.dx_model.predict(self.model, self.x_true)
        all_scores = self.dx_model.get_all_scores(self.y_true, y_pred, {})
        all_scores = [{'subject': '', **scores} for scores in all_scores]
        self.dx_model.customized_analysis(self.y_true, y_pred, all_scores)


# baseline model
def rnn_model(x_train, y_train, rnn_layer):
    main_input_shape = x_train['main_input'].shape[1:]
    dynamic_input = Input(shape=main_input_shape, name='main_input')
    x = Masking(mask_value=0.)(dynamic_input)
    x = GaussianNoise(0.02)(x)
    x = Bidirectional(rnn_layer(96, dropout=0, return_sequences=True))(x)
    x = Bidirectional(rnn_layer(30, dropout=0, return_sequences=True))(x)
    x = Bidirectional(rnn_layer(15, dropout=0, return_sequences=True))(x)
    aux_input_shape = x_train['aux_input'].shape[1:]
    aux_input = Input(shape=aux_input_shape, name='aux_input')
    x = concatenate([x, aux_input])
    x = Dense(15, use_bias=True, activation='tanh')(x)
    output_shape = y_train['output'].shape[2]
    output = Dense(output_shape, use_bias=True, name='output')(x)
    model = Model(inputs=[dynamic_input, aux_input], outputs=[output])

    # def custom_loss(y_true, y_pred):
    #     temp = K.cast(y_true >= 0, 'float32')
    #     return K.sum(K.abs(y_true - y_pred) * temp * y_true)

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
    x_fields = {'main_input': IMU_DATA_FIELDS, 'aux_input': [SUBJECT_WEIGHT, SUBJECT_HEIGHT]}
    y_fields = {'output': TARGETS_LIST}
    weights = {'output': [PHASE]*len(TARGETS_LIST)}
    dx_model = DXKamModel('40samples+stance_swing+padding_zero.h5', x_fields, y_fields, weights)
    # dx_model.preprocess_train_evaluation(range(11), range(11, 13), range(11, 13))
    subject_list = list(range(13))
    shuffle(subject_list)
    dx_model.cross_validation(subject_list)
