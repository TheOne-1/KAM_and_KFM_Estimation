import json
from typing import List
import h5py
import numpy as np
import prettytable as pt
from customized_logger import logger as logging
from const import SUBJECTS
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error as mse


class BaseModel:
    def __init__(self, data_path, x_fields, y_fields, weights=None, scalar=MinMaxScaler):
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
        self._scalars = {input_name: scalar() for input_name in list(x_fields.keys())+list(y_fields.keys())}
        with h5py.File(self._data_path, 'r') as hf:
            self._data_all_sub = {subject: hf[subject][:] for subject in SUBJECTS}
            self.data_columns = json.loads(hf.attrs['columns'])

    def _get_raw_data_dict(self, data, fields_dict):
        data_dict = {}
        for input_name, input_fields in fields_dict.items():
            x_field_col_loc = [self.data_columns.index(field_name) for field_name in input_fields]
            data_dict[input_name] = data[:, :, x_field_col_loc]
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
                mean_results += [{'output': output_name, 'field': field, 'r2': r2, 'rmse': rmse, 'mae': mae}]
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
        x_validation, y_validation = [None] * 2
        if validation_data is not None:
            x_validation = self._get_raw_data_dict(validation_data, self._x_fields)
            y_validation = self._get_raw_data_dict(validation_data, self._y_fields)
            x_validation, y_validation = self.preprocess_validation_test_data(x_validation, y_validation)
        model = self.train_model(x_train, y_train, x_validation, y_validation)
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

    def preprocess_train_data(self, x_train, y_train):
        for data_dict in [x_train, y_train]:
            for input_name, input_data in data_dict.items():
                original_shape = input_data.shape
                input_data = input_data.reshape([-1, input_data.shape[2]])
                input_data = self._scalars[input_name].fit_transform(input_data)
                input_data = input_data.reshape(original_shape)
                input_data[np.isnan(input_data)] = 0
                data_dict[input_name] = input_data
        return x_train, y_train

    def preprocess_validation_test_data(self, x, y):
        for data_dict in [x, y]:
            for input_name, input_data in data_dict.items():
                original_shape = input_data.shape
                input_data = input_data.reshape([-1, input_data.shape[2]])
                input_data = self._scalars[input_name].transform(input_data)
                input_data = input_data.reshape(original_shape)
                data_dict[input_name] = input_data
        return x, y

    @staticmethod
    def train_model(x_train, y_train, x_validation=None, y_validation=None):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, x_test):
        raise RuntimeError('Method not implemented')

    def get_all_scores(self, y_true, y_pred, weights):
        def get_colum_score(arr1, arr2, w=None):
            r2 = np.array([r2_score(arr1[i, :], arr2[i, :], sample_weight=None if w is None else w[i, :])
                           for i in range(arr2.shape[0])])
            rmse = np.array([np.sqrt(mse(arr1[i, :], arr2[i, :], sample_weight=None if w is None else w[i, :]))
                             for i in range(arr2.shape[0])])
            mae = np.array([np.average(abs((arr1[i, :] - arr2[i, :])), weights=None if w is None else w[i, :])
                            for i in range(arr2.shape[0])])
            return {'r2': r2, 'rmse': rmse, 'mae': mae}

        scores = []
        for output_name, fields in self._y_fields.items():
            for col, field in enumerate(fields):
                y_pred_one_field = y_pred[output_name][:, :, col]
                y_true_one_field = y_true[output_name][:, :, col]
                try:
                    weight_one_field = weights[output_name][:, :, col]
                except KeyError:
                    weight_one_field = None
                score_one_field = {'output': output_name, 'field': field}
                score_one_field.update(get_colum_score(y_pred_one_field, y_true_one_field, weight_one_field))
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
        axs[1, 1].text(0.4, 0.9, "mean r2: {}".format(np.round(np.mean(metric), 3)),
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
