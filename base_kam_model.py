import json
import h5py
import numpy as np
from const import SUBJECTS, VIDEO_DATA_FIELDS, IMU_DATA_FIELDS, TARGETS_LIST
from sklearn.metrics import r2_score, mean_squared_error
from typing import List
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BaseModel:
    def __init__(self, data_path, x_fields=IMU_DATA_FIELDS+VIDEO_DATA_FIELDS, y_fields=TARGETS_LIST,
                 scalar=MinMaxScaler):
        self._data_path = data_path
        self._x_fields = x_fields
        self._y_fields = y_fields
        self.scalar = scalar()
        with h5py.File(self._data_path, 'r') as hf:
            self._data_all_sub = {subject: hf[subject][:] for subject in SUBJECTS}
            self.data_columns = json.loads(hf.attrs['columns'])
            print(self.data_columns)

    def _depart_input_and_output(self, data):
        x_field_col_loc = [self.data_columns.index(field_name) for field_name in self._x_fields]
        y_field_col_loc = [self.data_columns.index(field_name) for field_name in self._y_fields]
        return data[:, :, x_field_col_loc], data[:, :, y_field_col_loc]

    def param_tuning(self, train_sub_ids: List[int], validate_sub_ids: List[int], test_sub_ids: List[int]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        train_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in train_sub_ids]
        train_data_list = [self._data_all_sub[sub_name] for sub_name in train_sub_names]

        validate_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in validate_sub_ids]
        validation_data_list = [self._data_all_sub[sub_name] for sub_name in validate_sub_names]

        test_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in test_sub_ids]
        test_data_list = [self._data_all_sub[sub_name] for sub_name in test_sub_names]

        print('Test the model with subjects: ' + str(test_sub_names)[1:-1])
        train_data = np.concatenate(train_data_list, axis=0)
        validation_data = np.concatenate(validation_data_list, axis=0)

        np.random.seed(0)
        np.random.shuffle(train_data)

        x_train, y_train = self._depart_input_and_output(train_data)
        x_train, y_train = self.preprocess_train_data(x_train, y_train)
        x_validation, y_validation = self._depart_input_and_output(validation_data)
        x_validation, y_validation = self.preprocess_validation_test_data(x_validation, y_validation)
        model = self.train_model(x_train, y_train, x_validation, y_validation)
        for test_sub_id, test_sub_name in enumerate(test_sub_names):
            test_sub_data = test_data_list[test_sub_id]
            test_sub_x, test_sub_y = self._depart_input_and_output(test_sub_data)
            test_sub_x, test_sub_y = self.preprocess_validation_test_data(test_sub_x, test_sub_y)
            pred_sub_y = self.predict(model, test_sub_x)
            test_results = {test_sub_name: self.get_all_scores(
                test_sub_y, pred_sub_y)}
            print(test_results)
            self.representative_profile_curves(test_sub_y, pred_sub_y, test_results)

    def preprocess_train_data(self, x_train, y_train):
        import copy
        x_ori = copy.deepcopy(x_train)

        original_shape = x_train.shape
        x_train = x_train.reshape([-1, x_train.shape[2]])
        x_train = self.scalar.fit_transform(x_train)
        x_train = x_train.reshape(original_shape)

        import matplotlib.pyplot as plt
        for axis in range(3, 6):
            plt.figure()
            plt.plot(x_ori[:, :, axis].ravel())
            plt.plot(x_train[:, :, axis].ravel())
        plt.show()

        return x_train, y_train

    def preprocess_validation_test_data(self, x, y):
        original_shape = x.shape
        x = x.reshape([-1, x.shape[2]])
        x = self.scalar.transform(x)
        x = x.reshape(original_shape)
        return x, y

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, x_test):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def get_all_scores(y_test, y_pred, num_of_digits=3):
        test, pred = np.ravel(y_test), np.ravel(y_pred)
        diffs = test - pred
        r2 = np.round(r2_score(test, pred), num_of_digits)
        rmse = np.round(np.sqrt(mean_squared_error(test, pred)), num_of_digits)
        mean_error = np.round(np.mean(diffs, axis=0), num_of_digits)
        absolute_mean_error = np.round(np.mean(abs(diffs)), num_of_digits)
        return r2, rmse, mean_error, absolute_mean_error

    @staticmethod
    def representative_profile_curves(y_test, y_pred, metrics):
        pass
