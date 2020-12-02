import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import Model, Input
from keras.layers import LSTM, GRU, Dense, Masking, Conv1D, Bidirectional, GaussianNoise, MaxPooling1D
from keras.layers import UpSampling1D, concatenate
from random import shuffle
import keras.backend as K
import keras.losses as Kloss
from sklearn.preprocessing import StandardScaler  # MinMaxScaler,
from keras.callbacks import Callback, ReduceLROnPlateau
from base_kam_model import BaseModel
from customized_logger import logger as logging
from const import DATA_PATH, SENSOR_LIST, VIDEO_LIST, SUBJECT_WEIGHT, SUBJECT_HEIGHT, PHASE

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DXKamModel(BaseModel):
    def __init__(self, data_file, input_fields, output_fields, output_weights):
        BaseModel.__init__(self, os.path.join(DATA_PATH, data_file), input_fields, output_fields, output_weights, StandardScaler)

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        model = rnn_model(x_train, y_train, GRU)
        validation_data = None
        callbacks = []
        if x_validation is not None:
            validation_data = (x_validation, y_validation)
            # callbacks.append(ErrorVisualization(self, x_validation, y_validation, validation_weight))
            callbacks.append(ReduceLROnPlateau('val_loss', factor=0.5, patience=5))
        model.fit(x=x_train, y=y_train, validation_data=validation_data, shuffle=True, batch_size=30,
                  epochs=20, verbose=0, callbacks=callbacks)
        return model

    @staticmethod
    def predict(model, x_test):
        prediction_list = model.predict(x_test, verbose=0)
        prediction_list = prediction_list if isinstance(prediction_list, list) else [prediction_list]
        prediction_dict = {name: pred for name, pred in zip(model.output_names, prediction_list)}
        return prediction_dict

    def get_all_scores(self, y_true, y_pred, weights=None):
        y_true = self.normalize_data(y_true, self._data_scalar, 'inverse_transform', 'by_each_column')
        y_pred = self.normalize_data(y_pred, self._data_scalar, 'inverse_transform', 'by_each_column')
        return BaseModel.get_all_scores(self, y_true, y_pred, weights)

    def preprocess_train_data(self, x, y):
        # KAM_index = self._y_fields['main_output'].index(RKAM_COLUMN)
        # height_index = self._x_fields['aux_input'].index(SUBJECT_HEIGHT)
        # y['main_output'][:, :, KAM_index] *= x['aux_input'][:, :, height_index]
        x1 = {'main_input_acc': x['main_input_acc'], 'main_input_gyr': x['main_input_gyr']}
        x2 = {'aux_input': x['aux_input']}
        x1 = self.normalize_data(x1, self._data_scalar, 'fit_transform', 'by_all_columns')
        x2 = self.normalize_data(x2, self._data_scalar, 'fit_transform', 'by_each_column')
        x = {**x1, **x2}
        y = self.normalize_data(y, self._data_scalar, 'fit_transform', 'by_each_column')
        return x, y

    def preprocess_validation_test_data(self, x, y):
        # KAM_index = self._y_fields['main_output'].index(RKAM_COLUMN)
        # height_index = self._x_fields['aux_input'].index(SUBJECT_HEIGHT)
        # y['main_output'][:, :, KAM_index] *= x['aux_input'][:, :, height_index]
        x1 = {'main_input_acc': x['main_input_acc'], 'main_input_gyr': x['main_input_gyr']}
        x2 = {'aux_input': x['aux_input']}
        x1 = self.normalize_data(x1, self._data_scalar, 'transform', 'by_all_columns')
        x2 = self.normalize_data(x2, self._data_scalar, 'transform', 'by_each_column')
        x = {**x1, **x2}
        y = self.normalize_data(y, self._data_scalar, 'transform', 'by_each_column')
        return x, y
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
    def __init__(self, model, x_true, y_true, validation_weight):
        Callback.__init__(self)
        self.dx_model = model
        self.x_true = x_true
        self.y_true = y_true
        self.validation_weight = validation_weight

    def on_epoch_begin(self, epoch, logs=None):
        y_pred = self.dx_model.predict(self.model, self.x_true)
        all_scores = self.dx_model.get_all_scores(self.y_true, y_pred, self.validation_weight)
        all_scores = [{'subject': '', **scores} for scores in all_scores]
        self.dx_model.customized_analysis(self.y_true, y_pred, all_scores)


print_model = True


# baseline model
def rnn_model(x_train, y_train, rnn_layer):
    main_input_acc_shape = x_train['main_input_acc'].shape[1:]
    main_input_gyr_shape = x_train['main_input_gyr'].shape[1:]
    main_input_acc = Input(shape=main_input_acc_shape, name='main_input_acc')
    main_input_gyr = Input(shape=main_input_gyr_shape, name='main_input_gyr')
    x = concatenate([main_input_acc, main_input_gyr])
    x = Masking(mask_value=0.)(x)
    # x = GaussianNoise(0.1)(x)
    x = Bidirectional(rnn_layer(90, dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional(rnn_layer(30, dropout=0.2, return_sequences=True))(x)
    x = Bidirectional(rnn_layer(15, dropout=0.2, return_sequences=True))(x)

    aux_input_shape = x_train['aux_input'].shape[1:]
    aux_input = Input(shape=aux_input_shape, name='aux_input')
    x = concatenate([x, aux_input])
    aux_output_shape = y_train['aux_output'].shape[2]
    aux_output = Dense(aux_output_shape, use_bias=True, name='aux_output')(x)
    x = concatenate([x, aux_output])
    x = Dense(15, use_bias=True)(x)
    output_shape = y_train['main_output'].shape[2]
    main_output = Dense(output_shape, use_bias=True, name='main_output')(x)
    model = Model(inputs=[main_input_acc, main_input_gyr, aux_input], outputs=[main_output, aux_output])

    def custom_loss(y_true, y_pred):
        temp = K.cast(y_true >= 0, 'float32')
        return K.sum(K.abs(y_true - y_pred) * (1. + K.log(temp * y_true + 1.)))

    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    global print_model
    if print_model:
        model.summary(print_fn=logging.info)
        print_model = False
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
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
    IMU_FIELDS_ACC = ['AccelX', 'AccelY', 'AccelZ']
    IMU_FIELDS_GYR = ['GyroX', 'GyroY', 'GyroZ']
    IMU_DATA_FIELDS_ACC = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_ACC]
    IMU_DATA_FIELDS_GYR = [IMU_FIELD + "_" + SENSOR for SENSOR in SENSOR_LIST for IMU_FIELD in IMU_FIELDS_GYR]
    VIDEO_DATA_FIELDS = [VIDEO + "_" + position + "_" + angle for VIDEO in VIDEO_LIST
                         for position in ["x", "y"] for angle in ["90", "180"]]

    x_fields = {'main_input_acc': IMU_DATA_FIELDS_ACC,
                'main_input_gyr': IMU_DATA_FIELDS_ACC,
                'aux_input':      [SUBJECT_WEIGHT, SUBJECT_HEIGHT]}
    # TARGETS_LIST = [RKAM_COLUMN]
    MAIN_TARGETS_LIST = ['RIGHT_KNEE_ADDUCTION_MOMENT', "RIGHT_KNEE_FLEXION_MOMENT"]
    AUX_TARGETS_LIST = ["RIGHT_KNEE_ADDUCTION_ANGLE", "RIGHT_KNEE_ADDUCTION_VELOCITY"]
    y_fields = {'main_output': MAIN_TARGETS_LIST, 'aux_output': AUX_TARGETS_LIST}
    y_weights = {'main_output': [PHASE] * len(MAIN_TARGETS_LIST), 'aux_output': [PHASE] * len(AUX_TARGETS_LIST)}
    dx_model = DXKamModel('40samples+stance_swing+baseline_only.h5', x_fields, y_fields, y_weights)
    subject_list = dx_model.get_all_subjects()
    shuffle(subject_list)
    # dx_model.preprocess_train_evaluation(subject_list[3:], subject_list[:3], subject_list[:3])
    dx_model.cross_validation(subject_list)
