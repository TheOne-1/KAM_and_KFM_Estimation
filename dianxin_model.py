import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras import Model, Input
from keras.layers import LSTM, GRU, Dense, Masking, Bidirectional, GaussianNoise
from keras.layers import concatenate
from random import shuffle
from sklearn.preprocessing import StandardScaler  # MinMaxScaler,
from keras.callbacks import Callback, ReduceLROnPlateau
from base_kam_model import BaseModel
from customized_logger import logger as logging
from const import DATA_PATH, SENSOR_LIST, SUBJECT_WEIGHT, SUBJECT_HEIGHT, KAM_PHASE, VIDEO_DATA_FIELDS
from const import extract_imu_fields, extract_right_force_fields, SEGMENT_DATA_FIELDS

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DXKamModel(BaseModel):
    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)
        self.print_model = True

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        model = rnn_model(x_train, y_train, GRU)
        if self.print_model:
            model.summary(print_fn=logging.info)
            self.print_model = False
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
        return BaseModel.get_all_scores(y_true, y_pred, self._y_fields, weights)

    def customized_analysis(self, sub_y_true, sub_y_pred, all_scores):
        sub_y_true = self.normalize_data(sub_y_true, self._data_scalar, 'inverse_transform', 'by_each_column')
        sub_y_pred = self.normalize_data(sub_y_pred, self._data_scalar, 'inverse_transform', 'by_each_column')
        return BaseModel.customized_analysis(self, sub_y_true, sub_y_pred, all_scores)

    def preprocess_train_data(self, x, y, weight):
        # KAM_index = self._y_fields['main_output'].index(R_KAM_COLUMN)
        # height_index = self._x_fields['aux_input'].index(SUBJECT_HEIGHT)
        # y['main_output'][:, :, KAM_index] *= x['aux_input'][:, :, height_index]
        x1 = {'main_input_acc': x['main_input_acc'], 'main_input_gyr': x['main_input_gyr']}
        x2 = {'aux_input': x['aux_input']}
        x1 = self.normalize_data(x1, self._data_scalar, 'fit_transform', 'by_each_column')
        x2 = self.normalize_data(x2, self._data_scalar, 'fit_transform', 'by_each_column')
        x = {**x1, **x2}
        y = self.normalize_data(y, self._data_scalar, 'fit_transform', 'by_each_column')
        return x, y, weight

    def preprocess_validation_test_data(self, x, y, weight):
        # KAM_index = self._y_fields['main_output'].index(R_KAM_COLUMN)
        # height_index = self._x_fields['aux_input'].index(SUBJECT_HEIGHT)
        # y['main_output'][:, :, KAM_index] *= x['aux_input'][:, :, height_index]
        x1 = {'main_input_acc': x['main_input_acc'], 'main_input_gyr': x['main_input_gyr']}
        x2 = {'aux_input': x['aux_input']}
        x1 = self.normalize_data(x1, self._data_scalar, 'transform', 'by_each_column')
        x2 = self.normalize_data(x2, self._data_scalar, 'transform', 'by_each_column')
        x = {**x1, **x2}
        y = self.normalize_data(y, self._data_scalar, 'transform', 'by_each_column')
        return x, y, weight


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


def rnn_model(x_train, y_train, rnn_layer):
    main_input_acc_shape = x_train['main_input_acc'].shape[1:]
    main_input_gyr_shape = x_train['main_input_gyr'].shape[1:]
    main_input_acc = Input(shape=main_input_acc_shape, name='main_input_acc')
    main_input_gyr = Input(shape=main_input_gyr_shape, name='main_input_gyr')
    x = concatenate([main_input_acc, main_input_gyr])
    x = Masking(mask_value=0.)(x)
    x = GaussianNoise(0.1)(x)
    x = rnn_layer(30, dropout=0.2, return_sequences=True)(x)
    x = rnn_layer(40, dropout=0.2, return_sequences=True)(x)

    aux_input_shape = x_train['aux_input'].shape[1:]
    aux_input = Input(shape=aux_input_shape, name='aux_input')
    x = concatenate([x, aux_input])
    aux_output_shape = y_train['aux_output'].shape[2]
    aux_output = Dense(aux_output_shape, use_bias=True, name='aux_output')(x)
    x = concatenate([x, aux_output])
    x = Dense(7, use_bias=True)(x)
    output_shape = y_train['main_output'].shape[2]
    main_output = Dense(output_shape, use_bias=True, name='main_output')(x)

    model = Model(inputs=[main_input_acc, main_input_gyr, aux_input], outputs=[main_output, aux_output])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


if __name__ == "__main__":
    # SENSOR_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']
    SENSOR_LIST = ['L_THIGH']
    ACC_FIELDS = extract_imu_fields(SENSOR_LIST, ['AccelX', 'AccelY', 'AccelZ'])
    GYR_FIELDS = extract_imu_fields(SENSOR_LIST, ['GyroX', 'GyroY', 'GyroZ'])
    FORCE_DATA = extract_right_force_fields(['force'], ['x', 'y', 'z'])

    MAIN_TARGETS_LIST = ['RIGHT_KNEE_ADDUCTION_MOMENT', "RIGHT_KNEE_FLEXION_MOMENT"]
    AUX_TARGETS_LIST = ["RIGHT_KNEE_ADDUCTION_VELOCITY"]

    x_fields = {'main_input_acc': ACC_FIELDS,
                'main_input_gyr': GYR_FIELDS,
                'aux_input':      [SUBJECT_WEIGHT, SUBJECT_HEIGHT]}
    y_fields = {'main_output': MAIN_TARGETS_LIST, 'aux_output': AUX_TARGETS_LIST}
    y_weights = {'main_output': [KAM_PHASE] * len(MAIN_TARGETS_LIST), 'aux_output': [KAM_PHASE] * len(AUX_TARGETS_LIST)}

    data_set = os.path.join(DATA_PATH, '40samples+stance_swing+padding_zero.h5')
    # data_set = os.path.join(DATA_PATH, 'stance_resampled.h5')
    dx_model = DXKamModel(data_set, x_fields, y_fields, y_weights, StandardScaler)
    subject_list = dx_model.get_all_subjects()
    # dx_model.preprocess_train_evaluation(subject_list[3:], subject_list[:3], subject_list[:3])
    dx_model.cross_validation(subject_list)
