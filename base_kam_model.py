import json
from typing import List
import h5py
import numpy as np
import prettytable as pt
from customized_logger import logger as logging
from const import SUBJECTS, VIDEO_DATA_FIELDS, IMU_DATA_FIELDS, TARGETS_LIST
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error


class BaseModel:
    def __init__(self, data_path, x_fields=IMU_DATA_FIELDS + VIDEO_DATA_FIELDS, y_fields=TARGETS_LIST,
                 scalar=MinMaxScaler):
        logging.info("Load data from h5 file {}".format(data_path))
        logging.info("Load data with input fields {}, output fields {}".format(x_fields, y_fields))
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
        sub_num = len(sub_ids)
        folder_num = int(np.ceil(sub_num / test_set_sub_num))  # the number of cross validation times
        results = []
        for i_folder in range(folder_num):
            test_sub_ids = sub_ids[test_set_sub_num * i_folder:test_set_sub_num * (i_folder + 1)]
            train_sub_ids = list(np.setdiff1d(sub_ids, test_sub_ids))
            test_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in test_sub_ids]
            logging.info('Cross validation: Subjects for test: {}'.format(test_sub_names))
            results += self.preprocess_train_evaluation(train_sub_ids, test_sub_ids, test_sub_ids)
        tb = pt.PrettyTable()
        for test_result in results:
            tb.field_names = test_result.keys()
            tb.add_row(test_result.values())
        print(tb)

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
        x_train, y_train = self._depart_input_and_output(train_data)
        x_train, y_train = self.preprocess_train_data(x_train, y_train)

        validate_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in validate_sub_ids]
        validation_data_list = [self._data_all_sub[sub_name] for sub_name in validate_sub_names]

        validation_data = np.concatenate(validation_data_list, axis=0) if validation_data_list else None
        x_validation, y_validation = [None] * 2
        if validation_data is not None:
            x_validation, y_validation = self._depart_input_and_output(validation_data)
            x_validation, y_validation = self.preprocess_validation_test_data(x_validation, y_validation)
        model = self.train_model(x_train, y_train, x_validation, y_validation)
        return model

    def model_evaluation(self, model, test_sub_ids: List[int]):
        test_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in test_sub_ids]
        test_data_list = [self._data_all_sub[sub_name] for sub_name in test_sub_names]
        test_results = []
        for test_sub_id, test_sub_name in enumerate(test_sub_names):
            test_sub_data = test_data_list[test_sub_id]
            test_sub_x, test_sub_y = self._depart_input_and_output(test_sub_data)
            test_sub_x, test_sub_y = self.preprocess_validation_test_data(test_sub_x, test_sub_y)
            pred_sub_y = self.predict(model, test_sub_x)
            sub_result = {'subject': test_sub_name}
            all_scores = self.get_all_scores(test_sub_y, pred_sub_y)
            self.customized_analysis(test_sub_y, pred_sub_y)
            for target_col, scores in enumerate(all_scores):
                target = self._y_fields[target_col]
                self.representative_profile_curves(test_sub_y[:, :, target_col], pred_sub_y[:, :, target_col],
                                                   "test result for {}, {}".format(test_sub_name, target),
                                                   scores['r2'])
                for score_name, score in scores.items():
                    sub_result['_'.join([target, score_name])] = np.round(np.mean(score), 3)
            test_results.append(sub_result)
        return test_results

    @staticmethod
    def customized_analysis(test_sub_y, pred_sub_y):
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
    def train_model(x_train, y_train, x_validation=None, y_validation=None):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, x_test):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def get_all_scores(y_true, y_pred, weight=None):
        scores = []
        for col in range(y_true.shape[2]):
            y_pred_one_target = y_pred[:, :, col]
            y_true_one_target = y_true[:, :, col]
            r2 = np.array([r2_score(y_pred_one_target[i, :], y_true_one_target[i, :],
                                    sample_weight=None if weight is None else weight[i, :])
                           for i in range(y_pred_one_target.shape[0])])
            rmse = np.array([np.sqrt(mean_squared_error(y_pred_one_target[i, :], y_true_one_target[i, :],
                                                        sample_weight=None if weight is None else weight[i, :]))
                             for i in range(y_pred_one_target.shape[0])])
            mae = np.array([np.average(abs((y_pred_one_target[i, :] - y_true_one_target[i, :])),
                                       weights=None if weight is None else weight[i, :])
                            for i in range(y_pred_one_target.shape[0])])
            scores.append({'r2': r2, 'rmse': rmse, 'mae': mae})
        return scores

    @staticmethod
    def representative_profile_curves(y_true, y_pred, title, metric):
        """
        y_true: 2-dimensional array
        y_pred: 2-dimensional array
        title: graph title
        metric: 1-dimensional array
        """
        # print r2 worst, median, best result
        best_index = metric.argmax()
        worst_index = metric.argmin()
        median_index = np.argsort(metric)[len(metric) // 2]
        fig, axs = plt.subplots(2, 2)
        plt.title = title
        for sub_index, [sub_title, step_index] in enumerate(
                [['r2_worst', worst_index], ['r2_mid', median_index], ['r2_best', best_index]]):
            axs[sub_index // 2, sub_index % 2].plot(y_true[step_index, :], 'g-', label='True_Value')
            axs[sub_index // 2, sub_index % 2].plot(y_pred[step_index, :], 'y-', label='Pred_Value')
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
        axs[1, 1].set_title('general performance')
        plt.tight_layout()
        plt.show(block=False)
