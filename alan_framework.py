""" This file records the code for major revision of TII """
import copy
import os
import random
from base_framework import BaseFramework
import torch
from customized_logger import logger as logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
import h5py
from const import IMU_FIELDS, SENSOR_LIST, DATA_PATH, VIDEO_LIST, SUBJECT_WEIGHT, FORCE_PHASE, RKNEE_MARKER_FIELDS, \
    FORCE_DATA_FIELDS, STATIC_DATA, SEGMENT_MASS_PERCENT, SUBJECT_ID, TRIAL_ID, LEVER_ARM_FIELDS, TRIALS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from types import SimpleNamespace
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials as HP_Trials
import warnings
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter


class InertialNet(nn.Module):
    def __init__(self, x_dim, seed=0, nlayer=2):
        super(InertialNet, self).__init__()
        torch.manual_seed(seed)
        self.rnn_layer = nn.LSTM(x_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.linear_1 = nn.Linear(2 * globals()['lstm_unit'], globals()['fcnn_unit'], bias=True)
        for layer in [self.linear_1]:
            nn.init.xavier_normal_(layer.weight)
        for name, param in self.rnn_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, sequence, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.rnn_layer(sequence)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=152)
        sequence = self.linear_1(lstm_out)
        return sequence


class VideoNet(InertialNet):
    pass


class FusionNet(nn.Module):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self, imu_subnet, vid_subnet, rank, nlayer=2):
        super(FusionNet, self).__init__()
        self.imu_subnet = imu_subnet
        self.vid_subnet = vid_subnet
        self.rank = rank
        self.fused_dim = 10

        self.imu_factor = Parameter(torch.Tensor(self.rank, 1, globals()['fcnn_unit'] + 1, self.fused_dim))
        self.vid_factor = Parameter(torch.Tensor(self.rank, 1, globals()['fcnn_unit'] + 1, self.fused_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim))

        # init factors
        nn.init.xavier_normal_(self.imu_factor)
        nn.init.xavier_normal_(self.vid_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        # additional LSTM layers
        self.rnn_after_fusion_layer = nn.LSTM(self.fused_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.linear_1 = nn.Linear(self.fused_dim, globals()['fcnn_unit'], bias=True)
        self.linear_2 = nn.Linear(globals()['fcnn_unit'], 2, bias=True)
        self.relu = nn.ReLU()

    def forward(self, imu_x, vid_x, anthro, lens):
        imu_h = self.imu_subnet(imu_x, lens)
        vid_h = self.vid_subnet(vid_x, lens)
        batch_size = imu_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        data_type = torch.cuda.FloatTensor

        _imu_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, imu_h.shape[1], 1).type(data_type), requires_grad=False), imu_h), dim=2)
        _vid_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, vid_x.shape[1], 1).type(data_type), requires_grad=False), vid_h), dim=2)

        fusion_imu = torch.matmul(_imu_h, self.imu_factor)
        fusion_vid = torch.matmul(_vid_h, self.vid_factor)
        fusion_zy = fusion_imu * fusion_vid

        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias

        # sequence1 = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        # lstm_out, _ = self.rnn_after_fusion_layer(sequence1)
        # lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=152)

        sequence = self.linear_1(sequence)
        sequence = self.relu(sequence)
        sequence = self.linear_2(sequence)

        weight = anthro[:, 0, 0].unsqueeze(1).unsqueeze(2)
        height = anthro[:, 0, 1].unsqueeze(1).unsqueeze(2)
        sequence = torch.div(sequence, weight * height)
        return sequence


class AlanFramework(BaseFramework):
    def __init__(self,  *args, **kwargs):
        BaseFramework.__init__(self, *args, **kwargs)
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3
        self.vid_static_cali()
        self.make_vid_vector_relative_to_midhip()
        self.get_body_weighted_imu()
        # self.add_additional_columns()

    def vid_static_cali(self):
        vid_y_90_col_loc = [self._data_fields.index(marker + '_y_90') for marker in VIDEO_LIST]
        for sub_name, sub_data in self._data_all_sub.items():
            static_side_df = pd.read_csv(DATA_PATH + '/' + sub_name + '/combined/static_side.csv', index_col=0)
            r_ankle_z = np.mean(static_side_df['RAnkle_y_90'])
            sub_data[:, :, vid_y_90_col_loc] = sub_data[:, :, vid_y_90_col_loc] - r_ankle_z + 1500
            self._data_all_sub[sub_name] = sub_data

    def make_vid_vector_relative_to_midhip(self):
        midhip_col_loc = [self._data_fields.index('MidHip' + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
        key_points_to_process = ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
        for sub_name, sub_data in self._data_all_sub.items():
            midhip_90_and_180_data = sub_data[:, :, midhip_col_loc]
            for key_point in key_points_to_process:
                key_point_col_loc = [self._data_fields.index(key_point + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
                sub_data[:, :, key_point_col_loc] = sub_data[:, :, key_point_col_loc] - midhip_90_and_180_data
            self._data_all_sub[sub_name] = sub_data

    def add_additional_columns(self):
        marker_rknee_col_loc = [self._data_fields.index(field_name) for field_name in RKNEE_MARKER_FIELDS]
        force_col_loc = [self._data_fields.index(field_name) for field_name in FORCE_DATA_FIELDS]
        for sub_name, sub_data in self._data_all_sub.items():
            marker_data = sub_data[:, :, marker_rknee_col_loc].copy()
            force_data = sub_data[:, :, force_col_loc].copy()
            knee_vector = force_data[:, :, 9:12] - (marker_data[:, :, :3] + marker_data[:, :, 3:6]) / 2
            self._data_all_sub[sub_name] = np.concatenate([sub_data, knee_vector], axis=2)
        self._data_fields.extend(LEVER_ARM_FIELDS)

    def get_body_weighted_imu(self):
        weight_col_loc = self._data_fields.index(SUBJECT_WEIGHT)
        for sub_name, sub_data in self._data_all_sub.items():
            sub_weight = sub_data[0, 0, weight_col_loc]
            for segment in SENSOR_LIST:
                segment_imu_col_loc = [self._data_fields.index(field + '_' + segment) for field in IMU_FIELDS[:6]]
                sub_data[:, :, segment_imu_col_loc[:3]] =\
                    sub_data[:, :, segment_imu_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT[segment] / 100

            self._data_all_sub[sub_name] = sub_data

    def preprocess_train_data(self, x, y, weight):
        self._x_fields_loc_and_mode = {}
        for k in set(list(x.keys())) - set(['anthro']):
            acc_loc = [self._x_fields[k].index(field) for field in self._x_fields[k] if 'Acc' in field]
            other_loc = [self._x_fields[k].index(field) for field in self._x_fields[k] if 'Acc' not in field]
            self._x_fields_loc_and_mode[k] = ([acc_loc, other_loc], ['_acc', '_other'], ['by_all_columns', 'by_each_column'])
            for loc, loc_name, mode in zip(*self._x_fields_loc_and_mode[k]):
                if len(loc) > 0:
                    x[k][:, :, loc] = self.normalize_array_separately(
                        x[k][:, :, loc], k+loc_name, 'fit_transform', scalar_mode=mode)
        return x, y, weight

    def preprocess_validation_test_data(self, x, y, weight):
        for k in set(list(x.keys())) - set(['anthro']):
            for loc, loc_name, mode in zip(*self._x_fields_loc_and_mode[k]):
                if len(loc) > 0:
                    x[k][:, :, loc] = self.normalize_array_separately(
                        x[k][:, :, loc], k+loc_name, 'transform', scalar_mode=mode)
        return x, y, weight

    def normalize_array_separately(self, data, name, method, scalar_mode='by_each_column'):
        if method == 'fit_transform':
            self._data_scalar[name] = MinMaxScaler(feature_range=(-3, 3))
        assert (scalar_mode in ['by_each_column', 'by_all_columns'])
        input_data = data.copy()
        original_shape = input_data.shape
        target_shape = [-1, input_data.shape[2]] if scalar_mode == 'by_each_column' else [-1, 1]
        input_data[(input_data == 0.).all(axis=2), :] = np.nan
        input_data = input_data.reshape(target_shape)
        scaled_data = getattr(self._data_scalar[name], method)(input_data)
        scaled_data = scaled_data.reshape(original_shape)
        scaled_data[np.isnan(scaled_data)] = 0.
        return scaled_data

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        def prepare_data(train_step_lens, validation_step_lens, batch_size):
            x_train_imu = torch.from_numpy(x_train['input_imu']).float().cuda()
            x_train_vid = torch.from_numpy(x_train['input_vid']).float().cuda()
            x_anthro = torch.from_numpy(x_train['anthro']).float().cuda()
            y_train_ = torch.from_numpy(y_train['main_output']).float().cuda()
            train_step_lens = torch.from_numpy(train_step_lens)
            train_ds = TensorDataset(x_train_imu, x_train_vid, x_anthro, y_train_, train_step_lens)
            train_size = int(0.9 * len(train_ds))
            vali_from_train_size = len(train_ds) - train_size
            train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds, [train_size, vali_from_train_size])
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=batch_size)

            x_validation_imu = torch.from_numpy(x_validation['input_imu']).float().cuda()
            x_validation_vid = torch.from_numpy(x_validation['input_vid']).float().cuda()
            x_anthro = torch.from_numpy(x_validation['anthro']).float().cuda()
            y_validation_ = torch.from_numpy(y_validation['main_output']).float().cuda()
            validation_step_lens = torch.from_numpy(validation_step_lens)
            test_ds = TensorDataset(x_validation_imu, x_validation_vid, x_anthro, y_validation_, validation_step_lens)
            test_dl = DataLoader(test_ds, batch_size=batch_size)
            vali_from_test_ds = TensorDataset(x_validation_imu, x_validation_vid, x_anthro, y_validation_, validation_step_lens)
            num_of_step_for_peek = int(0.3 * len(y_validation_))
            vali_from_test_ds, _ = torch.utils.data.dataset.random_split(vali_from_test_ds, [num_of_step_for_peek, len(
                y_validation_) - num_of_step_for_peek])
            vali_from_test_dl = DataLoader(vali_from_test_ds, batch_size=batch_size)
            return train_dl, vali_from_train_dl, vali_from_test_dl, test_dl

        def train(model, train_dl, optimizer, loss_fn, params):
            for i_batch, (xb_imu, xb_vid, xb_anthro, yb, lens) in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > params.use_ratio:
                    continue  # increase the speed of epoch
                optimizer.zero_grad()
                y_pred = model(xb_imu, xb_vid, xb_anthro, lens)
                train_loss = loss_fn(y_pred, yb)
                train_loss.backward()
                optimizer.step()

        def eval_after_training(model, test_dl, y_validation, validation_weight, params, show_plots=False):
            with torch.no_grad():
                y_pred_list = []
                for i_batch, (xb_imu, xb_vid, xb_anthro, yb, lens) in enumerate(test_dl):
                    y_pred_list.append(model(xb_imu, xb_vid, xb_anthro, lens).detach().cpu())
                y_pred = torch.cat(y_pred_list)
            y_pred = {params.target_name: y_pred.detach().cpu().numpy()}
            all_scores = BaseFramework.get_all_scores(y_validation, y_pred, {params.target_name: params.fields},
                                                      validation_weight)
            all_scores = [{'subject': 'all', **scores} for scores in all_scores]
            self.print_table(all_scores)
            if show_plots:
                self.customized_analysis(y_validation, y_pred, all_scores)
                plt.show()
            return y_pred

        def eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch):
            def vali_set_loss(nn_model, validation_dl, loss_fn):
                validation_loss = []
                for xb_imu, xb_vid, xb_anthro, yb, lens in validation_dl:
                    with torch.no_grad():
                        yb_pred = nn_model(xb_imu, xb_vid, xb_anthro, lens)
                        validation_loss.append(loss_fn(yb_pred, yb).item() / xb_imu.shape[0])
                return np.mean(validation_loss)

            vali_from_train_loss = vali_set_loss(model, vali_from_train_dl, loss_fn)
            vali_from_test_loss = vali_set_loss(model, vali_from_test_dl, loss_fn)
            logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t".format(
                i_epoch, vali_from_train_loss, vali_from_test_loss, time.time() - epoch_end_time))

        self.train_step_lens, self.validation_step_lens = self._get_step_len(x_train), self._get_step_len(x_validation)

        imu_subnet = InertialNet(x_train['input_imu'].shape[2], seed=0).cuda()
        vid_subnet = VideoNet(x_train['input_vid'].shape[2], seed=1).cuda()
        model = FusionNet(imu_subnet, vid_subnet, rank=10).cuda()

        hyper_param = {'epoch': globals()['epoch'], 'batch_size': globals()['batch_size'],
                       'lr': globals()['lr'], 'weight_decay': 0, 'use_ratio': 100}
        hyper_param.update({'target_name': 'main_output', 'fields': ['EXT_KM_X', 'EXT_KM_Y']})
        # self.build_main_model(model, x_train, y_train, x_validation, y_validation, validation_weight, hyper_param)
        params = SimpleNamespace(**hyper_param)
        train_dl, vali_from_train_dl, vali_from_test_dl, test_dl = prepare_data(
            self.train_step_lens, self.validation_step_lens, int(params.batch_size))
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        logging.info('\tEpoch | Validation_set_Loss | Test_set_Loss | Duration\t\t')
        epoch_end_time = time.time()
        for i_epoch in range(params.epoch):
            eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch)
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, loss_fn, params)
        eval_after_training(model, test_dl, y_validation, validation_weight, params)
        built_models = {'model': model}
        return built_models

    def predict(self, model, x_test):
        nn_model = model['model']
        self.test_step_lens = self._get_step_len(x_test)
        x_imu = torch.from_numpy(x_test['input_imu']).float().cuda()
        x_vid = torch.from_numpy(x_test['input_vid']).float().cuda()
        x_anthro = torch.from_numpy(x_test['anthro']).float().cuda()
        with torch.no_grad():
            test_ds = TensorDataset(x_imu, x_vid, x_anthro, torch.from_numpy(self.test_step_lens))
            test_dl = DataLoader(test_ds, batch_size=20)
            y_pred_list = []
            for i_batch, (xb_imu, xb_vid, xb_anthro, lens) in enumerate(test_dl):
                y_pred_list.append(nn_model(xb_imu, xb_vid, xb_anthro, lens).detach().cpu())
            y_pred = torch.cat(y_pred_list)
        y_pred = y_pred.detach().cpu().numpy()
        return {'main_output': y_pred}

    def save_model_and_results(self, test_sub_y, pred_sub_y, test_sub_weight, models, test_sub_name):
        save_path = os.path.join(self.result_dir, 'sub_models', test_sub_name)
        os.makedirs(save_path, exist_ok=True)
        for model_name, model in models.items():
            copied_model = copy.deepcopy(model)
            torch.save(copied_model.cpu(), os.path.join(save_path, model_name + '.pth'))

        results, columns = [], []
        for category, fields in self._y_fields.items():
            if len(fields) > 1:
                y_true_columns = fields
            else:
                y_true_columns = ['true_' + category]
            columns += y_true_columns
            results.append(test_sub_y[category])
        for category, fields_data in pred_sub_y.items():
            y_pred_columns = ['pred_' + category]
            columns += y_pred_columns
            results.append(fields_data)
        results = np.concatenate(results, axis=2)
        with h5py.File(os.path.join(self.result_dir, 'results.h5'), 'a') as hf:
            hf.require_dataset(test_sub_name, shape=results.shape, data=results, dtype='float32')
            hf.attrs['columns'] = json.dumps(columns)

    @staticmethod
    def _get_step_len(data, feature_col_num=0):
        """

        :param data: Numpy array, 3d (step, sample, feature)
        :param feature_col_num: int, feature column id for step length detection. Different id would probably return
               the same results
        :return:
        """
        data_the_feature = data[list(data.keys())[0]][:, :, feature_col_num]
        zero_loc = data_the_feature == 0.
        data_len = np.sum(~zero_loc, axis=1)
        return data_len

    def hyperparam_tuning(self, hyper_train_sub_ids, hyper_vali_sub_ids):
        logging.info('Searching best hyper parameters, subjects for validation: {}'.format(hyper_vali_sub_ids))
        logging.disabled = True
        global hyper_train_fun, hyper_vali_fun, hyper_train_ids, hyper_vali_ids
        hyper_train_fun, hyper_vali_fun = self.preprocess_and_train, self.model_evaluation
        hyper_train_ids, hyper_vali_ids = hyper_train_sub_ids, hyper_vali_sub_ids
        space = {
            'epoch': hp.quniform('epoch', 4, 10, 2),
            'lr': hp.uniform('lr', 10 ** -3, 10 ** -2),
            'batch_size': hp.quniform('batch_size', 10, 40, 10),
            'lstm_unit': hp.qnormal('lstm_unit', 40, 10, 1),
            'fcnn_unit': hp.qnormal('fcnn_unit', 40, 10, 1),
        }
        trials = HP_Trials()
        warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficent is not defined.")
        # best_param = fmin(objective_for_hyper_search, space, algo=tpe.suggest, max_evals=10, trials=trials,      # !!!
        #                   return_argmin=False, rstate=np.random.RandomState(seed=5))
        # show_hyper(trials, self.result_dir)
        best_param = {'epoch': 6, 'lr': 3e-3, 'batch_size': 32, 'lstm_unit': 15, 'fcnn_unit': 25}
        best_param = int_params(best_param)

        logging.disabled = False
        globals().update(best_param)
        best_param = {param: globals()[param] for param in ['epoch', 'lr', 'batch_size', 'weight_decay_1',
                                                            'lstm_unit', 'fcnn_unit'] if param in globals()}
        logging.info("Best hyper parameters: " + str(best_param))


def int_params(args):
    for arg_name in ['batch_size', 'epoch', 'fcnn_unit', 'lstm_unit']:
        if arg_name in args.keys():
            args[arg_name] = int(args[arg_name])
    return args


def show_hyper(trials, result_dir):
    save_path = os.path.join(DATA_PATH, 'training_results', result_dir, 'hyper_figure/')
    os.makedirs(save_path, exist_ok=True)
    for param_name in trials.trials[0]['misc']['vals'].keys():
        f, ax = plt.subplots(1)
        xs = [t['misc']['vals'][param_name] for t in trials.trials]
        ys = [t['result']['loss'] for t in trials.trials]
        ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
        if 'lr' in param_name:
            ax.set_xscale("log")
        ax.set_title(param_name, fontsize=18)
        ax.set_xlabel('$x$', fontsize=16)
        ax.set_ylabel('$val$', fontsize=16)
        plt.savefig(save_path+param_name+'.png')


def objective_for_hyper_search(args):
    args = int_params(args)
    print("Current: " + str(args), end='')
    globals().update(args)
    trained_model = hyper_train_fun(hyper_train_ids, hyper_vali_ids)
    hyper_search_results = hyper_vali_fun(trained_model, hyper_vali_ids, save_results=False)
    rmse_all = 0
    for element in hyper_search_results:
        rmse_all += element['rmse'].mean()
    print('RMSE = {}'.format(rmse_all / len(hyper_search_results)))
    return rmse_all / len(hyper_search_results)


def run(input_imu, input_vid, result_dir):
    x_fields = {'input_imu': input_imu, 'input_vid': input_vid}
    x_fields['anthro'] = STATIC_DATA
    y_fields = {
        'main_output': ['EXT_KM_X', 'EXT_KM_Y'],
        'auxiliary_info': [SUBJECT_ID, TRIAL_ID, FORCE_PHASE]
    }
    weights = {key: [FORCE_PHASE] * len(y_fields[key]) for key in y_fields.keys()}
    weights.update({key: [FORCE_PHASE] * len(x_fields[key]) for key in x_fields.keys()})
    evaluate_fields = {'main_output': y_fields['main_output']}

    model_builder = AlanFramework(data_path, x_fields, y_fields, TRIALS[:], weights, evaluate_fields,
                                  lambda: MinMaxScaler(feature_range=(-3, 3)), result_dir=result_dir)
    subjects = model_builder.get_all_subjects()
    # model_builder.preprocess_train_evaluation(subjects[:13], subjects[13:], subjects[13:])
    model_builder.cross_validation(subjects, 3)
    plt.close('all')

data_path = DATA_PATH + '/40samples+stance.h5'
VID_180_FIELDS = [loc + axis + '_180' for loc in ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"]
                  for axis in ['_x', '_y']]
VID_90_FIELDS = [loc + axis + '_90' for loc in ["LShoulder", "RShoulder", "RKnee", "LKnee", "RAnkle", "LAnkle"] for axis in ['_x', '_y']]

ACC_GYR_ALL = [field + '_' + sensor for sensor in SENSOR_LIST for field in IMU_FIELDS[:6]]
ACC_GYR_3 = [field + '_' + sensor for sensor in ['L_FOOT', 'R_FOOT', 'WAIST'] for field in IMU_FIELDS[:6]]
ACC_GYR_1 = [field + '_' + sensor for sensor in ['WAIST'] for field in IMU_FIELDS[:6]]

input_vid_2 = VID_90_FIELDS + VID_180_FIELDS
input_imu_8_all = ACC_GYR_ALL
# input_imu_3_all = {'force_x': ACC_GYR_3, 'force_y': ACC_GYR_3, 'force_z': ACC_GYR_3, 'r_x': ACC_GYR_3, 'r_y': ACC_GYR_3, 'r_z': R_FOOT_GYR}
# input_imu_1_all = {'force_x': ACC_GYR_1, 'force_y': ACC_GYR_1, 'force_z': ACC_GYR_1, 'r_x': ACC_GYR_1, 'r_y': ACC_GYR_1, 'r_z': ACC_GYR_1}

if __name__ == "__main__":
    """ Use all the IMU channels """
    result_date = '1016'
    run(input_imu=input_imu_8_all, input_vid=input_vid_2, result_dir=result_date + '_8IMU_2camera')


