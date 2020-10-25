import numpy as np
from config import DATA_PATH
import h5py
from const import IMU_DATA_FIELDS, VIDEO_DATA_FIELDS, TARGETS_LIST, SUBJECTS
from sklearn.metrics import r2_score, mean_squared_error
# CHANGES
# changed a typo trail to trial
# moved shared const to const.py

# QUESTIONS:
# event not in data, Evaluation need it


class BaseEvaluation:
    @staticmethod
    def _divide_data_to_each_sub(Y_test, Y_pred, test_sub_ids_of_each_step):
        sub_id_unique = list(np.unique(test_sub_ids_of_each_step))
        Y_test_sub, Y_pred_sub = {}, {}
        for sub_id in sub_id_unique:
            data_loc = np.where(test_sub_ids_of_each_step == sub_id)
            Y_test_sub[SUBJECTS[sub_id]] = Y_test[data_loc, :]
            Y_pred_sub[SUBJECTS[sub_id]] = Y_pred[data_loc, :]
        return Y_test_sub, Y_pred_sub, sub_id_unique

    @staticmethod
    def get_all_scores(Y_test, Y_pred, test_sub_ids_of_each_step, num_of_digits=3):
        Y_test_sub, Y_pred_sub, sub_id_unique = BaseEvaluation._divide_data_to_each_sub(
            Y_test, Y_pred, test_sub_ids_of_each_step)

        r2_list, rmse_list, mean_error_list, absolute_mean_error_list = [], [], [], []
        for sub_id in sub_id_unique:
            test, pred = Y_test_sub[SUBJECTS[sub_id]], Y_pred_sub[SUBJECTS[sub_id]]
            r2_list.append(r2_score(test, pred))
            rmse_list.append(np.sqrt(mean_squared_error(test, pred)))
            diffs = test - pred
            mean_error_list.append(np.mean(diffs, axis=0))
            absolute_mean_error_list.append(np.mean(abs(diffs)))

        r2 = np.round(np.mean(r2_list), num_of_digits)
        rmse = np.round(np.mean(rmse_list), num_of_digits)
        mean_error = np.round(np.mean(mean_error_list), num_of_digits)
        absolute_mean_erro = np.round(np.mean(absolute_mean_error_list), num_of_digits)
        return r2, rmse, mean_error, absolute_mean_erro

    @staticmethod
    def representative_profile_curves(Y_test, Y_pred, metrics):
        pass


class BaseModel:
    def __init__(self):
        self._data_path = DATA_PATH + 'whole_data_160.h5'
        with h5py.File(self._data_path, 'r') as hf:
            self._data_all_sub = {subject: hf[subject][:] for subject in SUBJECTS}

    def param_tuning(self, test_sub_ids):
        train_data_list, test_data_list = [], []
        test_sub_names = []
        test_sub_ids_of_each_step = np.zeros([0])
        for sub_name in SUBJECTS:
            sub_id = SUBJECTS.index(sub_name)
            if sub_id in test_sub_ids:
                test_data_list.append(self._data_all_sub[sub_name])
                test_sub_names.append(sub_name)
                id_of_each_step = np.full([self._data_all_sub[sub_name].shape[0]], sub_id)
                test_sub_ids_of_each_step = np.concatenate([test_sub_ids_of_each_step, id_of_each_step])
            else:
                train_data_list.append(self._data_all_sub[sub_name])
        print('Test the model with subjects: ' + str(test_sub_names)[1:-1])
        train_data = np.concatenate(train_data_list, axis=0)
        test_data = np.concatenate(test_data_list, axis=0)
        np.random.shuffle(train_data)
        train_data, test_data = self.preprocessing(train_data, test_data)
        X_train, Y_train, X_test, Y_test = train_data[:, :, 0:-1], train_data[:, :, -1:], \
                                           test_data[:, :, 0:-1], test_data[:, :, -1:]
        model = self.train_model(X_train, Y_train, X_test, Y_test)
        Y_pred = self.predict(model, X_test)
        r2, rmse, mean_error, absolute_mean_error = self.get_all_scores(
            Y_test, Y_pred, test_sub_ids_of_each_step)
        self.representative_profile_curves(Y_test, Y_pred, rmse)

    @staticmethod
    def preprocessing(train_data, test_data):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def train_model(X_train, Y_train, X_test, Y_test):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def predict(model, X_test):
        raise RuntimeError('Method not implemented')

    @staticmethod
    def _divide_data_to_each_sub(Y_test, Y_pred, test_sub_ids_of_each_step):
        sub_id_unique = list(np.unique(test_sub_ids_of_each_step))
        Y_test_sub, Y_pred_sub = {}, {}
        for sub_id in sub_id_unique:
            data_loc = np.where(test_sub_ids_of_each_step == sub_id)
            Y_test_sub[SUBJECTS[sub_id]] = Y_test[data_loc, :]
            Y_pred_sub[SUBJECTS[sub_id]] = Y_pred[data_loc, :]
        return Y_test_sub, Y_pred_sub, sub_id_unique

    @staticmethod
    def get_all_scores(Y_test, Y_pred, test_sub_ids_of_each_step, num_of_digits=3):
        Y_test_sub, Y_pred_sub, sub_id_unique = BaseEvaluation._divide_data_to_each_sub(
            Y_test, Y_pred, test_sub_ids_of_each_step)

        r2_list, rmse_list, mean_error_list, absolute_mean_error_list = [], [], [], []
        for sub_id in sub_id_unique:
            test, pred = Y_test_sub[SUBJECTS[sub_id]], Y_pred_sub[SUBJECTS[sub_id]]
            r2_list.append(r2_score(test, pred))
            rmse_list.append(np.sqrt(mean_squared_error(test, pred)))
            diffs = test - pred
            mean_error_list.append(np.mean(diffs, axis=0))
            absolute_mean_error_list.append(np.mean(abs(diffs)))

        r2 = np.round(np.mean(r2_list), num_of_digits)
        rmse = np.round(np.mean(rmse_list), num_of_digits)
        mean_error = np.round(np.mean(mean_error_list), num_of_digits)
        absolute_mean_erro = np.round(np.mean(absolute_mean_error_list), num_of_digits)
        return r2, rmse, mean_error, absolute_mean_erro

    @staticmethod
    def representative_profile_curves(Y_test, Y_pred, metrics):
        pass
