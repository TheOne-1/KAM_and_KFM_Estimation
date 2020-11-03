import os
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
    def __init__(self):
        BaseModel.__init__(self, os.path.join(DATA_PATH, '40samples+stance_swing+padding_next_step.h5'),
                           IMU_DATA_FIELDS + VIDEO_DATA_FIELDS, TARGETS_LIST)

    def train_model(self, x_train, y_train, x_validation, y_validation):
        # TODO: feed IMU data fields and Video data fields into two separated network at the beginning.
        shape = x_train.shape[1:]
        model = gru_model(shape)

        # You might see this WARNING:
        # Allocation of 1823325480 exceeds 10% of free system memory.
        # This is because you batch size is so large. Try to use 20 or similar.
        model.fit(x_train, y_train, batch_size=20, epochs=40, verbose=1,
                  callbacks=[ErrorVisualization(x_validation, y_validation)])
        return model

    def predict(self, model, y_test):
        return model.predict(y_test, verbose=1)


class ErrorVisualization(Callback):
    def __init__(self, x_test, y_test):
        Callback.__init__(self)
        self.X_test = x_test
        self.Y_test = y_test

    def on_train_begin(self, logs=None):
        y_test = self.Y_test
        y_pred = self.model.predict(self.X_test)
        BaseModel.representative_profile_curves(y_test, y_pred, {})

    def on_epoch_end(self, epoch, logs=None):
        y_test = self.Y_test
        y_pred = self.model.predict(self.X_test)
        BaseModel.representative_profile_curves(y_test, y_pred, {})


def gru_model(input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    # You might see this WARNING:
    # tensorflow:Layer gru will not use cuDNN kernel since it doesn't meet the cuDNN kernel criteria.
    # It will use generic GPU kernel as fallback when running on GPU
    # There is requirement for GRU layer:
    # 1. activation == tanh
    # 2. recurrent_activation == sigmoid
    # 3. recurrent_dropout == 0
    # 4. unroll is False
    # 5. use_bias is True
    # 6. reset_after is True
    # 7. Inputs, if use masking, are strictly right_padded
    # 8. Eager execution is enabled in the outer most context
    model.add(GRU(20, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(30, dropout=0.2, return_sequences=True))
    model.add(Dense(5))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model


if __name__ == "__main__":
    GOOD_SUBJECTS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    dx_model = DXKamModel()
    dx_model.param_tuning(GOOD_SUBJECTS[0:-1], GOOD_SUBJECTS[-1:], GOOD_SUBJECTS[-1:])
