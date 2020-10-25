import numpy as np
from config import DATA_PATH
import h5py
from const import SUBJECTS
from sklearn.metrics import r2_score, mean_squared_error


class BaseModel:
    def __init__(self):
        self._data_path = DATA_PATH + 'whole_data_160.h5'
        with h5py.File(self._data_path, 'r') as hf:
            self._data_all_sub = {subject: hf[subject][:] for subject in SUBJECTS}

    @staticmethod
    def _depart_input_and_output(data):
        return data[:, :, 0:-1], data[:, :, -1:]

    def param_tuning(self, train_sub_ids, validate_sub_ids, test_sub_ids):
        train_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in train_sub_ids]
        train_data_list = [self._data_all_sub[sub_name] for sub_name in train_sub_names]

        validate_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in validate_sub_ids]
        validate_data_list = [self._data_all_sub[sub_name] for sub_name in validate_sub_names]

        test_sub_names = [sub_name for sub_index, sub_name in enumerate(SUBJECTS) if sub_index in test_sub_ids]
        test_data_list = [self._data_all_sub[sub_name] for sub_name in test_sub_names]

        print('Test the model with subjects: ' + str(test_sub_names)[1:-1])
        train_data_list, validate_data_list, test_data_list = self.preprocessing(train_data_list, validate_data_list,
                                                                                 test_data_list)
        train_data = np.concatenate(train_data_list, axis=0)
        validation_data = np.concatenate(validate_data_list, axis=0)

        np.random.shuffle(train_data)

        x_train, y_train = self._depart_input_and_output(train_data)
        x_validation, y_validation = self._depart_input_and_output(validation_data)
        model = self.train_model(x_train, y_train, x_validation, y_validation)
        for test_sub_id, test_sub_name in enumerate(test_sub_names):
            test_sub_data = test_data_list[test_sub_id]
            test_sub_x, test_sub_y = self._depart_input_and_output(test_sub_data)
            pred_sub_y = self.predict(model, test_sub_x)
            test_results = {test_sub_name: self.get_all_scores(
                test_sub_y, pred_sub_y)}
            print(test_results)
            self.representative_profile_curves(test_sub_y, pred_sub_y, test_results)

    @staticmethod
    def preprocessing(train_data, validation_data, test_data):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def train_model(x_train, y_train, x_test, y_test):
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
