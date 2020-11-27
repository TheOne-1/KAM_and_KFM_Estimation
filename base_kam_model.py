import json
from typing import List
import h5py
import numpy as np
import prettytable as pt
from customized_logger import logger as logging
from const import SUBJECTS, SENSOR_LIST, DATA_PATH, MODAL_FIELDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
import pandas as pd
from transforms3d.euler import euler2mat
from wearable_toolkit import DataScalar

"""To wdx: """
CALI_VIA_GRAVITY = False     # by default, static_back is used for calibration
INPUT_NORM_EACH_MODAL = True            # if true, norm each modal, if false, norm each channel
# import DivideMaxScalar in dianxin_model and use it


class BaseModel:
    def __init__(self, data_path, x_fields, y_fields, weights=None, base_scalar=MinMaxScaler):
        """
        x_fileds: a dict contains input names and input fields
        y_fileds: a dict contains output names and output fields
        """
        logging.info("Load data from h5 file {}".format(data_path))
        logging.info("Load data with input fields {}, output fields {}".format(x_fields, y_fields))
        self._data_path = data_path
        self._x_fields = x_fields
        self._y_fields = y_fields
        self._weights = {} if weights is None else weights
        self._data_scalar = DataScalar(base_scalar, x_fields, y_fields, INPUT_NORM_EACH_MODAL)
        with h5py.File(self._data_path, 'r') as hf:
            self._data_all_sub = {subject: hf[subject][:] for subject in SUBJECTS}
            self.data_columns = json.loads(hf.attrs['columns'])
        if CALI_VIA_GRAVITY:
            self.cali_via_gravity()

    def _get_raw_data_dict(self, data, fields_dict):
        data_dict = {}
        for input_name, input_fields in fields_dict.items():
            x_field_col_loc = [self.data_columns.index(field_name) for field_name in input_fields]
            data_dict[input_name] = data[:, :, x_field_col_loc].copy()
        return data_dict

    def preprocess_train_evaluation(self, train_sub_ids: List[int], validate_sub_ids: List[int],
                                    test_sub_ids: List[int]):
        logging.debug('Train the model with subject ids: {}'.format(train_sub_ids))
        logging.debug('Test the model with subject ids: {}'.format(test_sub_ids))
        logging.debug('Validate the model with subject ids: {}'.format(validate_sub_ids))
        model = self.preprocess_and_train(train_sub_ids, validate_sub_ids)
        test_results = self.model_evaluation(model, test_sub_ids)
        return test_results

    def cross_validation(self, sub_ids: List[int], test_set_sub_num=1):
        logging.info('Cross validation with subject ids: {}'.format(sub_ids))
        folder_num = int(np.ceil(len(sub_ids) / test_set_sub_num))  # the number of cross validation times
        results = []
        for i_folder in range(folder_num):
            test_sub_ids = sub_ids[test_set_sub_num * i_folder:test_set_sub_num * (i_folder + 1)]
            train_sub_ids = list(np.setdiff1d(sub_ids, test_sub_ids))
            test_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in test_sub_ids]
            logging.info('Cross validation: Subjects for test: {}'.format(test_sub_names))
            results += self.preprocess_train_evaluation(train_sub_ids, test_sub_ids, test_sub_ids)
        self.print_table(results)
        # get mean results
        mean_results = []
        for output_name, output_fields in self._y_fields.items():
            for field in output_fields:
                field_results = list(filter(lambda x: x['output'] == output_name and x['field'] == field, results))
                r2 = np.round(np.mean(np.concatenate([res['r2'] for res in field_results])), 3)
                rmse = np.round(np.mean(np.concatenate([res['rmse'] for res in field_results])), 3)
                mae = np.round(np.mean(np.concatenate([res['mae'] for res in field_results])), 3)
                r_rmse = np.round(np.mean(np.concatenate([res['r_rmse'] for res in field_results])), 3)
                cor_value = np.round(np.mean(np.concatenate([res['cor_value'] for res in field_results])), 3)
                mean_results += [{'output': output_name, 'field': field, 'r2': r2, 'rmse': rmse, 'mae': mae,
                                  'r_rmse': r_rmse, 'cor_value': cor_value}]
        self.print_table(mean_results)

    def preprocess_and_train(self, train_sub_ids: List[int], validate_sub_ids: List[int]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        train_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in train_sub_ids]
        train_data_list = [self._data_all_sub[sub_name] for sub_name in train_sub_names]

        train_data = np.concatenate(train_data_list, axis=0)
        train_data = shuffle(train_data, random_state=0)
        x_train = self._get_raw_data_dict(train_data, self._x_fields)
        y_train = self._get_raw_data_dict(train_data, self._y_fields)
        x_train, y_train = self.preprocess_train_data(x_train, y_train)

        validate_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in validate_sub_ids]
        validation_data_list = [self._data_all_sub[sub_name] for sub_name in validate_sub_names]

        validation_data = np.concatenate(validation_data_list, axis=0) if validation_data_list else None
        x_validation, y_validation, validation_weight = [None] * 3
        if validation_data is not None:
            x_validation = self._get_raw_data_dict(validation_data, self._x_fields)
            y_validation = self._get_raw_data_dict(validation_data, self._y_fields)
            validation_weight = self._get_raw_data_dict(validation_data, self._weights)
            x_validation, y_validation = self.preprocess_validation_test_data(x_validation, y_validation)
        model = self.train_model(x_train, y_train, x_validation, y_validation, validation_weight)
        return model

    def model_evaluation(self, model, test_sub_ids: List[int]):
        test_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in test_sub_ids]
        test_data_list = [self._data_all_sub[sub_name] for sub_name in test_sub_names]
        test_results = []
        for test_sub_id, test_sub_name in enumerate(test_sub_names):
            test_sub_data = test_data_list[test_sub_id]
            test_sub_x = self._get_raw_data_dict(test_sub_data, self._x_fields)
            test_sub_y = self._get_raw_data_dict(test_sub_data, self._y_fields)
            test_sub_weight = self._get_raw_data_dict(test_sub_data, self._weights)
            test_sub_x, test_sub_y = self.preprocess_validation_test_data(test_sub_x, test_sub_y)
            pred_sub_y = self.predict(model, test_sub_x)
            all_scores = self.get_all_scores(test_sub_y, pred_sub_y, test_sub_weight)
            all_scores = [{'subject': test_sub_name, **scores} for scores in all_scores]
            self.customized_analysis(test_sub_y, pred_sub_y, all_scores)
            test_results += all_scores
        self.print_table(test_results)
        return test_results

    def customized_analysis(self, test_sub_y, pred_sub_y, all_scores):
        """ Customized data visualization"""
        for score in all_scores:
            subject, output, field, r2 = [score[f] for f in ['subject', 'output', 'field', 'r2']]
            field_index = self._y_fields[output].index(field)
            arr1 = test_sub_y[output][:, :, field_index]
            arr2 = pred_sub_y[output][:, :, field_index]
            title = "{}, {}, {}, r2".format(subject, output, field, 'r2')
            self.representative_profile_curves(arr1, arr2, title, r2)

    def preprocess_train_data(self, x, y):
        self._data_scalar.scale_train_set(x)
        return x, y

    def preprocess_validation_test_data(self, x, y):
        self._data_scalar.scale_validation_test_set(x)
        return x, y

    @staticmethod
    def train_model(x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, x_test):
        raise RuntimeError('Method not implemented')

    def get_all_scores(self, y_true, y_pred, weights=None):
        def get_column_score(arr_true, arr_pred, w=None):
            r2, rmse, mae, r_rmse, cor_value = [np.zeros(arr_true.shape[0]) for _ in range(5)]
            for i in range(arr_true.shape[0]):
                arr_true_i = arr_true[i, :] if w is None else arr_true[i, w[i, :] == 1.]
                arr_pred_i = arr_pred[i, :] if w is None else arr_pred[i, w[i, :] == 1.]

                r2[i] = r2_score(arr_true_i, arr_pred_i)
                rmse[i] = np.sqrt(mse(arr_true_i, arr_pred_i))
                mae[i] = np.mean(abs((arr_true_i - arr_pred_i)))
                r_rmse[i] = rmse[i] / (arr_true_i.max() + arr_pred_i.max() - arr_true_i.min() - arr_pred_i.min()) / 2
                cor_value[i] = pearsonr(arr_true_i, arr_pred_i)[0]

            locs = np.where(w.ravel() == 1.)[0]
            r2_all = r2_score(arr_true.ravel()[locs], arr_pred.ravel()[locs])
            r2_all = np.full(r2.shape, r2_all)
            return {'r2': r2, 'r2_all': r2_all, 'rmse': rmse, 'mae': mae, 'r_rmse': r_rmse,
                    'cor_value': cor_value}

        scores = []
        weights = {} if weights is None else weights
        for output_name, fields in self._y_fields.items():
            for col, field in enumerate(fields):
                y_true_one_field = y_true[output_name][:, :, col]
                y_pred_one_field = y_pred[output_name][:, :, col]
                try:
                    weight_one_field = weights[output_name][:, :, col]
                except KeyError:
                    weight_one_field = None
                score_one_field = {'output': output_name, 'field': field}
                score_one_field.update(get_column_score(y_true_one_field, y_pred_one_field, weight_one_field))
                scores.append(score_one_field)
        return scores

    @staticmethod
    def representative_profile_curves(arr1, arr2, title, metric):
        """
        arr1: 2-dimensional array
        arr2: 2-dimensional array
        title: graph title
        metric: 1-dimensional array
        """
        # print r2 worst, median, best result
        best_index = metric.argmax()
        worst_index = metric.argmin()
        median_index = np.argsort(metric)[len(metric) // 2]
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(title)
        for sub_index, [sub_title, step_index] in enumerate(
                [['r2_worst', worst_index], ['r2_median', median_index], ['r2_best', best_index]]):
            sub_ax = axs[sub_index // 2, sub_index % 2]
            sub_ax.plot(arr1[step_index, :], 'g-', label='True_Value')
            sub_ax.plot(arr2[step_index, :], 'y-', label='Pred_Value')
            sub_ax.legend(loc='upper right', fontsize=8)
            sub_ax.text(0.4, 0.9, "r2: {}".format(np.round(metric[step_index], 3)), horizontalalignment='center',
                        verticalalignment='center', transform=sub_ax.transAxes)
            sub_ax.set_title(sub_title)

        # plot the general prediction status result
        y_pred_mean = arr2.mean(axis=0).reshape((-1))
        y_pred_std = arr2.std(axis=0).reshape((-1))
        y_true_mean = arr1.mean(axis=0)
        y_true_std = arr1.std(axis=0)
        axis_x = range(y_true_mean.shape[0])
        axs[1, 1].plot(axis_x, y_true_mean, 'g-', label='Real_Value')
        axs[1, 1].fill_between(axis_x, y_true_mean - y_true_std, y_true_mean + y_true_std, facecolor='green',
                               alpha=0.2)
        axs[1, 1].plot(axis_x, y_pred_mean, 'y-', label='Predict_Value')
        axs[1, 1].fill_between(axis_x, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std,
                               facecolor='yellow', alpha=0.2)
        axs[1, 1].legend(loc='upper right', fontsize=8)
        axs[1, 1].text(0.4, 0.9, "r2: {}".format(np.round(np.mean(metric), 3)),
                       horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('general performance')
        plt.tight_layout()
        plt.show(block=False)

    @staticmethod
    def print_table(results):
        tb = pt.PrettyTable()
        for test_result in results:
            tb.field_names = test_result.keys()
            tb.add_row([np.round(np.mean(value), 3) if isinstance(value, np.ndarray) else value
                        for value in test_result.values()])
        print(tb)

    def cali_via_gravity(self):
        for subject in SUBJECTS:
            logging.info("Rotating {}'s data".format(subject))
            transform_mat_dict = self.get_static_calibration(subject)
            for sensor in SENSOR_LIST:
                transform_mat = transform_mat_dict[sensor]
                rotation_fun = lambda data: np.matmul(transform_mat, data)
                acc_cols = ['Accel' + axis + sensor for axis in ['X_', 'Y_', 'Z_']]
                acc_col_locs = [self.data_columns.index(col) for col in acc_cols]
                self._data_all_sub[subject][:, :, acc_col_locs] = np.apply_along_axis(
                    rotation_fun, 2, self._data_all_sub[subject][:, :, acc_col_locs])
                gyr_cols = ['Gyro' + axis + sensor for axis in ['X_', 'Y_', 'Z_']]
                gyr_col_locs = [self.data_columns.index(col) for col in gyr_cols]
                self._data_all_sub[subject][:, :, gyr_col_locs] = np.apply_along_axis(
                    rotation_fun, 2, self._data_all_sub[subject][:, :, gyr_col_locs])

    @staticmethod
    def get_static_calibration(subject_name, trial_name='static_back'):
        data_path = DATA_PATH + '/' + subject_name + '/combined/' + trial_name + '.csv'
        static_data = pd.read_csv(data_path, index_col=0)
        transform_mat_dict = {}
        for sensor in SENSOR_LIST:
            acc_cols = ['Accel' + axis + sensor for axis in ['X_', 'Y_', 'Z_']]
            acc_mean = np.mean(static_data[acc_cols], axis=0)
            roll = np.arctan2(acc_mean[1], acc_mean[2])
            pitch = np.arctan2(-acc_mean[0], np.sqrt(acc_mean[1] ** 2 + acc_mean[2] ** 2))
            # print(np.rad2deg(roll), np.rad2deg(pitch))
            transform_mat_dict[sensor] = euler2mat(roll, pitch, 0)
        return transform_mat_dict
