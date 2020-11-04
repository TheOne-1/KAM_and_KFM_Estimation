import json
import h5py
import numpy as np
from customized_logger import logger as logging
from const import SUBJECTS, VIDEO_DATA_FIELDS, IMU_DATA_FIELDS, TARGETS_LIST
from sklearn.metrics import r2_score, mean_squared_error
from typing import List
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BaseModel:
    def __init__(self, data_path, x_fields=IMU_DATA_FIELDS + VIDEO_DATA_FIELDS, y_fields=TARGETS_LIST,
                 scalar=MinMaxScaler):
        logging.info("Create Model from h5 file {}".format(data_path))
        logging.info("Create Model with input fields {}, output ields {}".format(x_fields, y_fields))
        self._data_path = data_path
        self._x_fields = x_fields
        self._y_fields = y_fields
        self.scalar = scalar()
        with h5py.File(self._data_path, 'r') as hf:
            self._data_all_sub = {subject: hf[subject][:] for subject in SUBJECTS}
            self.data_columns = json.loads(hf.attrs['columns'])

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

        logging.info('Train the model with subjects: {}'.format(train_sub_names))
        logging.info('Validate the model with subjects: {}'.format(validate_sub_names))
        logging.info('Test the model with subjects: {}'.format(test_sub_names))
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
            logging.info("for now, the final test result is {}".format(test_results))
            self.representative_profile_curves(test_sub_y, pred_sub_y, test_results)
            self.customized_analysis(test_sub_y, pred_sub_y, test_results)
        plt.show()

    @staticmethod
    def customized_analysis(test_sub_y, pred_sub_y, test_results):
        """ Customized data visualization"""
        pass

    def preprocess_train_data(self, x_train, y_train):
        original_shape = x_train.shape
        x_train = x_train.reshape([-1, x_train.shape[2]])
        x_train = self.scalar.fit_transform(x_train)
        x_train = x_train.reshape(original_shape)
        x_train[np.isnan(x_train)] = 0
        y_train[np.isnan(y_train)] = 0
        return x_train, y_train

    def preprocess_validation_test_data(self, x, y):
        original_shape = x.shape
        x = x.reshape([-1, x.shape[2]])
        x = self.scalar.transform(x)
        x = x.reshape(original_shape)
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0
        return x, y

    @staticmethod
    def train_model(x_train, y_train, x_validation, y_validation):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, x_test):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def get_all_scores(y_true, y_pred, num_of_digits=3):
        y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
        y_pred = y_pred[y_true != 0.]  # remove those padding zeros
        y_true = y_true[y_true != 0.]  # remove those padding zeros
        diffs = y_true - y_pred
        r2 = np.round(r2_score(y_true, y_pred), num_of_digits)
        rmse = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), num_of_digits)
        mean_error = np.round(np.mean(diffs, axis=0), num_of_digits)
        mean_absolute_error = np.round(np.mean(abs(diffs)), num_of_digits)
        scores = {'r2': r2, 'rmse': rmse, 'mean_error': mean_error, 'mean_absolute_error': mean_absolute_error}
        return scores

    @staticmethod
    def representative_profile_curves(y_true, y_pred, metrics):
        y_pred = y_pred[:, :, 0]
        y_true = y_true[:, :, 0]
        axis_x = range(y_true.shape[1])
        diff = y_pred - y_true
        mean_error = np.mean(diff, axis=1)
        rmse = np.mean(diff ** 2, axis=1) ** 0.5
        r2 = np.array([r2_score(y_pred[i, y_true[i, :] != 0], y_true[i, y_true[i, :] != 0])  # y_pred[i, :] != 0 kick padding zeroes
                       for i in range(y_pred.shape[0])])
        mean_absolute_error = np.mean(abs(diff), axis=1)

        # print r2 worst, median, best result
        r2_best_index = r2.argmax()
        r2_worst_index = r2.argmin()
        r2_mid_index = np.argsort(r2)[len(r2) // 2]
        fig, axs = plt.subplots(2, 2)
        for sub_index, [sub_title, step_index] in enumerate(
                [['r2_worst', r2_worst_index], ['r2_mid', r2_mid_index], ['r2_best', r2_best_index]]):
            axs[sub_index // 2, sub_index % 2].plot(axis_x, y_true[step_index, :], 'g-', label='True_Value')
            axs[sub_index // 2, sub_index % 2].plot(axis_x, y_pred[step_index, :], 'y-', label='Pred_Value')
            axs[sub_index // 2, sub_index % 2].legend(loc='upper right', fontsize=8)
            axs[sub_index // 2, sub_index % 2].set_title(sub_title)

        # plot the general prediction status result
        y_pred_mean = y_pred.mean(axis=0).reshape((-1))
        y_pred_std = y_pred.std(axis=0).reshape((-1))
        y_true_mean = y_true.mean(axis=0)
        y_true_std = y_true.std(axis=0)
        axis_x = range(y_true_mean.shape[0])
        axs[1, 1].plot(axis_x, y_true_mean, 'g-', label='Real_Value')
        axs[1, 1].fill_between(axis_x, y_true_mean - y_true_std, y_true_mean + y_true_std, facecolor='green',
                               alpha=0.2)
        axs[1, 1].plot(axis_x, y_pred_mean, 'y-', label='Predict_Value')
        axs[1, 1].fill_between(axis_x, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std,
                               facecolor='yellow', alpha=0.2)
        axs[1, 1].legend(loc='upper right', fontsize=8)
        axs[1, 1].set_title("General Predicting status")
        plt.tight_layout()
        plt.show(block=False)

