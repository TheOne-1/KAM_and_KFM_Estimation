import os
import json
import h5py
import datetime
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
from typing import List
from customized_logger import logger as logging, add_file_handler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
import scipy.interpolate as interpo
from const import DATA_PATH, TRIALS
import random


def execute_cmd(cmd):
    return os.popen(cmd).read()


class BaseFramework:
    def __init__(self, data_path, x_fields, y_fields, specify_trials=None, weights=None, evaluate_fields=None,
                 base_scalar=MinMaxScaler, result_dir=None):
        """
        x_fileds: a dict contains input names and input fields
        y_fileds: a dict contains output names and output fields
        """
        # log to file
        if result_dir is not None:
            self.result_dir = os.path.join(DATA_PATH, 'training_results', result_dir)
        else:
            self.result_dir = os.path.join(DATA_PATH, 'training_results', str(datetime.datetime.now()))
        if evaluate_fields is None:
            self._evaluate_fields = y_fields
        else:
            self._evaluate_fields = evaluate_fields
        os.makedirs(os.path.join(self.result_dir, 'sub_figs'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'sub_models'), exist_ok=True)
        add_file_handler(logging, os.path.join(self.result_dir, 'training_log'))
        logging.info("Current commit is {}".format(execute_cmd("git rev-parse HEAD")))
        logging.info("Load data from h5 file {}".format(data_path))
        logging.info("Load data with input fields {}, output fields {}".format(x_fields, y_fields))
        self._x_fields = x_fields
        self._y_fields = y_fields
        self._weights = {} if weights is None else weights
        self._base_scalar = base_scalar
        self._data_scalar = {input_name: base_scalar() for input_name in list(x_fields.keys()) + list(y_fields.keys())}
        with h5py.File(data_path, 'r') as hf:
            self._data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
            self._data_fields = json.loads(hf.attrs['columns'])
            if specify_trials:
                trial_id_col_loc = self._data_fields.index('trial_id')
                specify_trial_ids = [TRIALS.index(specify_trial) for specify_trial in specify_trials]
                for subject, subject_data in self._data_all_sub.items():
                    self._data_all_sub[subject] = np.concatenate(
                        [subject_data[subject_data[:, 0, trial_id_col_loc] == specific_id, :, :] for specific_id in specify_trial_ids], axis=0)

    def get_all_subjects(self):
        return list(self._data_all_sub.keys())

    def _get_raw_data_dict(self, data, fields_dict):
        data_dict = {}
        for input_name, input_fields in fields_dict.items():
            x_field_col_loc = [self._data_fields.index(field_name) for field_name in input_fields]
            data_dict[input_name] = data[:, :, x_field_col_loc].copy()
        return data_dict

    def preprocess_train_evaluation(self, train_sub_ids: List[str], validate_sub_ids: List[str],
                                    test_sub_ids: List[str]):
        logging.debug('Train the model with subject ids: {}'.format(train_sub_ids))
        logging.debug('Test the model with subject ids: {}'.format(test_sub_ids))
        logging.debug('Validate the model with subject ids: {}'.format(validate_sub_ids))
        model = self.preprocess_and_train(train_sub_ids, validate_sub_ids)
        test_results = self.model_evaluation(model, test_sub_ids)
        plt.close('all')
        return test_results

    def cross_validation(self, sub_ids: List[str], test_set_sub_num=1):
        sub_ids = shuffle(sub_ids, random_state=0)
        logging.info('Cross validation with subject ids: {}'.format(sub_ids))
        folder_num = int(np.floor(len(sub_ids) / test_set_sub_num))  # the number of cross validation times
        results = []
        for i_folder in range(folder_num):
            logging.info('Current folder: {}'.format(i_folder))
            if i_folder < folder_num - 1:
                test_sub_ids = sub_ids[test_set_sub_num * i_folder:test_set_sub_num * (i_folder + 1)]
            else:
                test_sub_ids = sub_ids[test_set_sub_num * i_folder:]    # make use of all the left subjects
            train_sub_ids = list(np.setdiff1d(sub_ids, test_sub_ids))

            random.seed(0)
            hyper_vali_sub_ids = random.sample(train_sub_ids, test_set_sub_num)
            hyper_train_sub_ids = list(np.setdiff1d(train_sub_ids, hyper_vali_sub_ids))
            self.hyperparam_tuning(hyper_train_sub_ids, hyper_vali_sub_ids)

            logging.info('Cross validation, subjects for test: {}'.format(test_sub_ids))
            results += self.preprocess_train_evaluation(train_sub_ids, test_sub_ids, test_sub_ids)
        results = sorted(results, key=lambda x: (x['output'], x['field'], np.mean(x['r_rmse'])))
        self.print_table(results)
        # get mean results
        mean_results = []
        for output_name, output_fields in self._evaluate_fields.items():
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
        return mean_results

    def preprocess_and_train(self, train_sub_ids: List[str], validate_sub_ids: List[str]):
        """
        train_sub_ids: a list of subject id for model training
        validate_sub_ids: a list of subject id for model validation
        test_sub_ids: a list of subject id for model testing
        """
        train_data_list = [self._data_all_sub[sub_name] for sub_name in train_sub_ids]
        train_data = np.concatenate(train_data_list, axis=0)
        train_data = shuffle(train_data, random_state=0)
        x_train = self._get_raw_data_dict(train_data, self._x_fields)
        y_train = self._get_raw_data_dict(train_data, self._y_fields)
        train_weight = self._get_raw_data_dict(train_data, self._weights)
        x_train, y_train, train_weight = self.preprocess_train_data(x_train, y_train, train_weight)

        validation_data_list = [self._data_all_sub[sub] for sub in validate_sub_ids]
        validation_data = np.concatenate(validation_data_list, axis=0) if validation_data_list else None
        validation_data = shuffle(validation_data, random_state=0)
        x_validation, y_validation, validation_weight = [None] * 3
        if validation_data is not None:
            x_validation = self._get_raw_data_dict(validation_data, self._x_fields)
            y_validation = self._get_raw_data_dict(validation_data, self._y_fields)
            validation_weight = self._get_raw_data_dict(validation_data, self._weights)
            x_validation, y_validation, validation_weight = self.preprocess_validation_test_data(x_validation,
                                                                                                 y_validation,
                                                                                                 validation_weight)

        model = self.train_model(x_train, y_train, x_validation, y_validation, validation_weight)
        return model

    def model_evaluation(self, model, test_sub_ids: List[str], save_results=True):
        test_data_list = [self._data_all_sub[sub] for sub in test_sub_ids]
        test_results = []
        for test_sub_id, test_sub_name in enumerate(test_sub_ids):
            test_sub_data = test_data_list[test_sub_id]
            test_sub_x = self._get_raw_data_dict(test_sub_data, self._x_fields)
            test_sub_y = self._get_raw_data_dict(test_sub_data, self._y_fields)
            test_sub_weight = self._get_raw_data_dict(test_sub_data, self._weights)
            test_sub_x, test_sub_y, test_sub_weight = self.preprocess_validation_test_data(test_sub_x, test_sub_y,
                                                                                           test_sub_weight)
            pred_sub_y = self.predict(model, test_sub_x)
            all_scores = self.get_all_scores(test_sub_y, pred_sub_y, self._evaluate_fields, test_sub_weight)
            all_scores = [{'subject': test_sub_name, **scores} for scores in all_scores]
            self.customized_analysis(test_sub_y, pred_sub_y, all_scores)
            if save_results:
                self.save_model_and_results(test_sub_y, pred_sub_y, test_sub_weight, model, test_sub_name)
            test_results += all_scores
        self.print_table(test_results)
        return test_results

    @staticmethod
    def save_model_and_results(test_sub_y, pred_sub_y, test_sub_weight, model, test_sub_name):
        pass

    def customized_analysis(self, sub_y_true, sub_y_pred, all_scores):
        """ Customized data visualization"""
        for score in all_scores:
            subject, output, field, r_rmse = [score[f] for f in ['subject', 'output', 'field', 'r_rmse']]
            field_index = self._y_fields[output].index(field)
            arr1 = sub_y_true[output][:, :, field_index]
            arr2 = sub_y_pred[output][:, :, field_index]
            title = "{}, {}, {}, r_rmse".format(subject, output, field, 'r_rmse')
            self.representative_profile_curves(arr1, arr2, title, r_rmse)

    @staticmethod
    def normalize_data(data, scalars, method, scalar_mode='by_each_column'):
        assert (scalar_mode in ['by_each_column', 'by_all_columns'])
        scaled_data = {}
        for input_name, input_data in data.items():
            input_data = input_data.copy()
            original_shape = input_data.shape
            target_shape = [-1, input_data.shape[2]] if scalar_mode == 'by_each_column' else [-1, 1]
            input_data[(input_data == 0.).all(axis=2), :] = np.nan
            input_data = input_data.reshape(target_shape)
            input_data = getattr(scalars[input_name], method)(input_data)
            input_data = input_data.reshape(original_shape)
            input_data[np.isnan(input_data)] = 0.
            scaled_data[input_name] = input_data
        return scaled_data

    def preprocess_train_data(self, x, y, weight):
        x = self.normalize_data(x, self._data_scalar, 'fit_transform')
        for output_name, fields in self._y_fields.items():
            y[output_name], weight[output_name] = self.keep_stance_then_resample(y[output_name], weight[output_name])
        return x, y, weight

    def preprocess_validation_test_data(self, x, y, weight):
        x = self.normalize_data(x, self._data_scalar, 'transform')
        for output_name, fields in self._y_fields.items():
            y[output_name], weight[output_name] = self.keep_stance_then_resample(y[output_name], weight[output_name])
        return x, y, weight

    @staticmethod
    def keep_stance_then_resample(y, weight, resampled_len=100):
        y_resampled = np.zeros([y.shape[0], resampled_len, y.shape[2]])
        for i_output in range(y.shape[2]):
            for j_row in range(y.shape[0]):
                data_array = y[j_row, np.abs(weight[j_row, :, i_output] - 0) > 1e-5, i_output]
                y_resampled[j_row, :, i_output] = BaseFramework.resample_one_array(data_array, resampled_len)
        return y_resampled, np.full(y_resampled.shape, 1)

    @staticmethod
    def resample_one_array(data_array, resampled_len):
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        data_len = data_array.shape[1]
        data_step = np.arange(0, data_len)
        resampled_step = np.linspace(0, data_len - 1 + 1e-10, resampled_len)
        tck, data_step = interpo.splprep(data_array, u=data_step, s=0)
        data_resampled = interpo.splev(resampled_step, tck, der=0)[0]
        return data_resampled

    @staticmethod
    def train_model(x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, x_test):
        raise RuntimeError('Method not implemented')

    def hyperparam_tuning(self, model, x_test):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def get_all_scores(y_true, y_pred, y_fields, weights=None):
        def get_column_score(arr_true, arr_pred, w):
            r2, rmse, mae, r_rmse, cor_value = [np.zeros(arr_true.shape[0]) for _ in range(5)]
            for i in range(arr_true.shape[0]):
                arr_true_i = arr_true[i, w[i, :]]
                arr_pred_i = arr_pred[i, w[i, :]]

                r2[i] = r2_score(arr_true_i, arr_pred_i)
                rmse[i] = np.sqrt(mse(arr_true_i, arr_pred_i))
                mae[i] = np.mean(abs((arr_true_i - arr_pred_i)))
                r_rmse[i] = rmse[i] / (arr_true.max() - arr_true.min())
                cor_value[i] = pearsonr(arr_true_i, arr_pred_i)[0]

            locs = np.where(w.ravel())[0]
            r2_all = r2_score(arr_true.ravel()[locs], arr_pred.ravel()[locs])
            r2_all = np.full(r2.shape, r2_all)
            return {'r2': r2, 'r2_all': r2_all, 'rmse': rmse, 'mae': mae, 'r_rmse': r_rmse, 'cor_value': cor_value}

        scores = []
        weights = {} if weights is None else weights
        for output_name, fields in y_fields.items():
            for col, field in enumerate(fields):
                y_true_one_field = y_true[output_name][:, :, col]
                y_pred_one_field = y_pred[output_name][:, :, col]
                try:
                    weight_one_field = weights[output_name][:, :, col] == 1.
                except KeyError:
                    weight_one_field = np.full(y_true_one_field.shape, True)
                    logging.warning("Use default all true value for {}".format(output_name))
                score_one_field = {'output': output_name, 'field': field}
                score_one_field.update(get_column_score(y_true_one_field, y_pred_one_field, weight_one_field))
                scores.append(score_one_field)
        return scores

    def representative_profile_curves(self, arr_true, arr_pred, title, metric):
        """
        arr1: 2-dimensional array
        arr2: 2-dimensional array
        title: graph title
        metric: 1-dimensional array
        """
        # print metric worst, median, best result
        high_index, low_index, median_index = metric.argmax(), metric.argmin(), np.argsort(metric)[len(metric) // 2]

        fig, axs = plt.subplots(2, 2)
        fig.suptitle(title)
        for sub_index, [sub_title, step_index] in enumerate(
                [['low', low_index], ['median', median_index], ['high', high_index]]):
            sub_ax = axs[sub_index // 2, sub_index % 2]
            sub_ax.plot(arr_true[step_index, :], color='green', label='True_Value')
            sub_ax.plot(arr_pred[step_index, :], color='peru', label='Pred_Value')
            sub_ax.legend(loc='upper right', fontsize=8)
            sub_ax.text(0.4, 0.9, np.round(metric[step_index], 3), transform=sub_ax.transAxes)
            sub_ax.set_title(sub_title)

        # plot the general prediction status result
        arr_true_mean, arr_true_std = arr_true.mean(axis=0), arr_true.std(axis=0)
        arr_pred_mean, arr_pred_std = arr_pred.mean(axis=0), arr_pred.std(axis=0)
        axis_x = range(arr_true_mean.shape[0])
        axs[1, 1].plot(axis_x, arr_true_mean, color='green', label='Real_Value')
        axs[1, 1].fill_between(axis_x, arr_true_mean - arr_true_std, arr_true_mean + arr_true_std,
                               facecolor='green', alpha=0.3)
        axs[1, 1].plot(axis_x, arr_pred_mean, color='peru', label='Predict_Value')
        axs[1, 1].fill_between(axis_x, arr_pred_mean - arr_pred_std, arr_pred_mean + arr_pred_std,
                               facecolor='peru', alpha=0.3)
        axs[1, 1].legend(loc='upper right', fontsize=8)
        axs[1, 1].text(0.4, 0.9, np.round(np.mean(metric), 3), transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('general performance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'sub_figs', title))
        plt.close()

    @staticmethod
    def print_table(results):
        tb = pt.PrettyTable()
        for test_result in results:
            tb.field_names = test_result.keys()
            tb.add_row([np.round(np.mean(value), 3) if isinstance(value, np.ndarray) else value
                        for value in test_result.values()])
        logging.info(tb)
