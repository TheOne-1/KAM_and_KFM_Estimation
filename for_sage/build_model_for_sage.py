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
    FORCE_DATA_FIELDS, STATIC_DATA, SEGMENT_MASS_PERCENT, SUBJECT_ID, TRIAL_ID, GRAVITY, TRIALS, \
    SUBJECT_HEIGHT, USED_KEYPOINTS, HIGH_LEVEL_FEATURE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from types import SimpleNamespace
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.parameter import Parameter


class InertialNet(nn.Module):
    def __init__(self, x_dim, net_name, seed=0, nlayer=1):
        super(InertialNet, self).__init__()
        self.net_name = net_name
        torch.manual_seed(seed)
        self.rnn_layer = nn.LSTM(x_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        for name, param in self.rnn_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def __str__(self):
        return self.net_name

    def forward(self, sequence, lens):
        sequence = pack_padded_sequence(sequence, lens, batch_first=True, enforce_sorted=False)
        sequence, _ = self.rnn_layer(sequence)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=152)
        sequence = self.dropout(sequence)
        return sequence


class VideoNet(InertialNet):
    pass


class OutNet(nn.Module):
    def __init__(self, input_dim, high_level_locs=[2, 3, 4]):
        super(OutNet, self).__init__()
        self.high_level_locs = high_level_locs
        self.linear_1 = nn.Linear(input_dim + len(high_level_locs), globals()['fcnn_unit'], bias=True)
        self.linear_2 = nn.Linear(globals()['fcnn_unit'], 2, bias=True)
        self.relu = nn.ReLU()
        for layer in [self.linear_1, self.linear_2]:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, sequence, others):
        if len(self.high_level_locs) > 0:
            sequence = torch.cat((sequence, others[:, :, self.high_level_locs]), dim=2)
        sequence = self.linear_1(sequence)
        sequence = self.relu(sequence)
        sequence = self.linear_2(sequence)
        weight = others[:, 0, WEIGHT].unsqueeze(1).unsqueeze(2)
        height = others[:, 0, HEIGHT].unsqueeze(1).unsqueeze(2)
        sequence = torch.div(sequence, weight * GRAVITY * height / 100)
        return sequence


class LmfImuOnlyNet(nn.Module):
    """ Implemented based on the paper "Efficient low-rank multimodal fusion with modality-specific factors" """
    def __init__(self, acc_dim, gyr_dim):
        super(LmfImuOnlyNet, self).__init__()
        self.acc_subnet = InertialNet(acc_dim, 'acc net', seed=0)
        self.gyr_subnet = InertialNet(gyr_dim, 'gyr net', seed=0)
        self.vid_subnet = VideoNet(24, 'vid net', seed=0)
        self.rank = 10
        self.fused_dim = 40

        self.acc_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
        self.gyr_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
        self.vid_factor = Parameter(torch.Tensor(self.rank, 1, 2*globals()['lstm_unit'] + 1, self.fused_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.fused_dim))

        # init factors
        nn.init.xavier_normal_(self.acc_factor, 10)
        nn.init.xavier_normal_(self.gyr_factor, 10)
        nn.init.xavier_normal_(self.vid_factor, 10)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        if acc_dim <= 3:
            self.out_net = OutNet(self.fused_dim, [])  # do not use high level features
        else:
            self.out_net = OutNet(self.fused_dim, [2])  # only use FPA from high level features

    def __str__(self):
        return 'LMF IMU only net'

    def set_scalars(self, scalars):
        self.scalars = scalars

    def forward(self, acc_x, gyr_x, vid_x, others, lens):
        acc_h = self.acc_subnet(acc_x, lens)
        gyr_h = self.gyr_subnet(gyr_x, lens)
        batch_size = acc_h.data.shape[0]
        data_type = torch.cuda.FloatTensor

        _acc_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, acc_h.shape[1], 1).type(data_type), requires_grad=False), acc_h), dim=2)
        _gyr_h = torch.cat((torch.autograd.Variable(torch.ones(batch_size, gyr_h.shape[1], 1).type(data_type), requires_grad=False), gyr_h), dim=2)

        fusion_acc = torch.matmul(_acc_h, self.acc_factor)
        fusion_gyr = torch.matmul(_gyr_h, self.gyr_factor)
        fusion_vid = torch.full_like(fusion_acc, 1)
        fusion_zy = fusion_acc * fusion_gyr * fusion_vid
        # permute to make batch first
        sequence = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 2, 0, 3)).squeeze(dim=2) + self.fusion_bias
        sequence = self.out_net(sequence, others)
        return sequence


class AlanFramework(BaseFramework):
    def __init__(self,  *args, **kwargs):
        BaseFramework.__init__(self, *args, **kwargs)
        self.train_step_lens, self.validation_step_lens, self.test_step_lens = [None] * 3
        # self.add_additional_columns()
        self.make_vid_relative_to_midhip()
        self.normalize_vid_by_size_of_subject_in_static_trial()
        # self.get_body_weighted_imu()

    @staticmethod
    def angle_between_vectors(v1, v2):
        data_len = v1.shape[0]
        v1, v2 = v1.reshape([-1, 2]), v2.reshape([-1, 2])
        v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
        v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
        sign = np.sign(v1_u - v2_u)[:, 0]
        angle = sign * np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))
        return angle.reshape([data_len, -1, 1])

    def add_additional_columns(self):
        for sub_name, sub_data in self._data_all_sub.items():
            angles = []
            for vid in ['_90', '_180']:
                hip_loc, knee_loc, ankle_loc = [[self._data_fields.index(marker + '_x' + vid), self._data_fields.index(marker + '_y' + vid)]
                                                for marker in ["RHip", "RKnee", "RAnkle"]]
                vector_r_shank = sub_data[:, :, knee_loc] - sub_data[:, :, ankle_loc]
                vector_r_thigh = sub_data[:, :, hip_loc] - sub_data[:, :, knee_loc]
                angles.append(self.angle_between_vectors(vector_r_shank, vector_r_thigh))
            self._data_all_sub[sub_name] = np.concatenate([sub_data, angles[0], angles[1]], axis=2)
        self._data_fields.extend(['r_knee_angle_90', 'r_knee_angle_180'])

    def vid_static_cali(self):
        vid_y_90_col_loc = [self._data_fields.index(marker + '_y_90') for marker in VIDEO_LIST]
        for sub_name, sub_data in self._data_all_sub.items():
            static_side_df = pd.read_csv(DATA_PATH + '/' + sub_name + '/combined/static_side.csv', index_col=0)
            r_ankle_z = np.mean(static_side_df['RAnkle_y_90'])
            sub_data[:, :, vid_y_90_col_loc] = sub_data[:, :, vid_y_90_col_loc] - r_ankle_z + 1500
            self._data_all_sub[sub_name] = sub_data

    def make_vid_relative_to_midhip(self):
        midhip_col_loc = [self._data_fields.index('MidHip' + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
        for sub_name, sub_data in self._data_all_sub.items():
            midhip_90_and_180_data = sub_data[:, :, midhip_col_loc]
            for key_point in USED_KEYPOINTS:
                key_point_col_loc = [self._data_fields.index(key_point + axis + angle) for axis in ['_x', '_y'] for angle in ['_90', '_180']]
                sub_data[:, :, key_point_col_loc] = sub_data[:, :, key_point_col_loc] - midhip_90_and_180_data
            self._data_all_sub[sub_name] = sub_data

    def normalize_vid_by_size_of_subject_in_static_trial(self):
        for sub_name, sub_data in self._data_all_sub.items():
            height_col_loc = self._data_fields.index(SUBJECT_HEIGHT)
            sub_height = sub_data[0, 0, height_col_loc]
            for camera in ['90', '180']:
                vid_col_loc = [self._data_fields.index(keypoint + axis + camera) for keypoint in USED_KEYPOINTS for axis in ['_x_', '_y_']]
                sub_data[:, :, vid_col_loc] = sub_data[:, :, vid_col_loc] / sub_height
            self._data_all_sub[sub_name] = sub_data

    def get_body_weighted_imu(self):
        weight_col_loc = self._data_fields.index(SUBJECT_WEIGHT)
        for sub_name, sub_data in self._data_all_sub.items():
            sub_weight = sub_data[0, 0, weight_col_loc]
            for segment in SENSOR_LIST:
                segment_imu_col_loc = [self._data_fields.index(field + '_' + segment) for field in IMU_FIELDS[:6]]
                sub_data[:, :, segment_imu_col_loc[:3]] = \
                    sub_data[:, :, segment_imu_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT[segment] / 100

            self._data_all_sub[sub_name] = sub_data

    def preprocess_train_data(self, x, y, weight):
        for k in set(list(x.keys())) - set(['anthro']):
            x[k] = self.normalize_array_separately(x[k], k, 'fit_transform')
        return x, y, weight

    def preprocess_validation_test_data(self, x, y, weight):
        for k in set(list(x.keys())) - set(['anthro']):
            x[k] = self.normalize_array_separately(x[k], k, 'transform')
        return x, y, weight

    def normalize_array_separately(self, data, name, method, scalar_mode='by_each_column'):
        if method == 'fit_transform':
            self._data_scalar[name] = MinMaxScaler(feature_range=(-1, 1))      # MinMaxScaler(feature_range=(-1, 1)) StandardScaler()
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
            x_train_acc = torch.from_numpy(x_train['input_acc']).float().cuda()
            x_train_gyr = torch.from_numpy(x_train['input_gyr']).float().cuda()
            x_train_vid = torch.from_numpy(x_train['input_vid']).float().cuda()
            x_train_others = np.concatenate([x_train['anthro'], x_train['high_level']], axis=2)
            x_train_others = torch.from_numpy(x_train_others).float().cuda()
            y_train_ = torch.from_numpy(y_train['main_output']).float().cuda()
            train_step_lens = torch.from_numpy(train_step_lens)
            train_ds = TensorDataset(x_train_acc, x_train_gyr, x_train_vid, x_train_others, y_train_, train_step_lens)
            train_size = int(0.96 * len(train_ds))
            vali_from_train_size = len(train_ds) - train_size
            train_ds, vali_from_train_ds = torch.utils.data.dataset.random_split(train_ds, [train_size, vali_from_train_size])
            train_dl = DataLoader(train_ds, batch_size=batch_size)
            vali_from_train_dl = DataLoader(vali_from_train_ds, batch_size=batch_size)

            x_validation_acc = torch.from_numpy(x_validation['input_acc']).float().cuda()
            x_validation_gyr = torch.from_numpy(x_validation['input_gyr']).float().cuda()
            x_validation_vid = torch.from_numpy(x_validation['input_vid']).float().cuda()
            x_vali_others = np.concatenate([x_validation['anthro'], x_validation['high_level']], axis=2)
            x_vali_others = torch.from_numpy(x_vali_others).float().cuda()
            y_validation_ = torch.from_numpy(y_validation['main_output']).float().cuda()
            validation_step_lens = torch.from_numpy(validation_step_lens)
            test_ds = TensorDataset(x_validation_acc, x_validation_gyr, x_validation_vid, x_vali_others, y_validation_, validation_step_lens)
            test_dl = DataLoader(test_ds, batch_size=batch_size)
            vali_from_test_ds = TensorDataset(x_validation_acc, x_validation_gyr, x_validation_vid, x_vali_others, y_validation_, validation_step_lens)
            num_of_step_for_peek = int(0.3 * len(y_validation_))
            vali_from_test_ds, _ = torch.utils.data.dataset.random_split(vali_from_test_ds, [num_of_step_for_peek, len(
                y_validation_) - num_of_step_for_peek])
            vali_from_test_dl = DataLoader(vali_from_test_ds, batch_size=batch_size)
            return train_dl, vali_from_train_dl, vali_from_test_dl, test_dl

        def train(model, train_dl, optimizer, loss_fn, params):
            model.train()
            for i_batch, (xb_acc, xb_gyr, xb_vid, xb_others, yb, lens) in enumerate(train_dl):
                n = random.randint(1, 100)
                if n > params.use_ratio:
                    continue  # increase the speed of epoch
                optimizer.zero_grad()
                y_pred = model(xb_acc, xb_gyr, xb_vid, xb_others, lens)
                loss_fn(y_pred, yb).backward()
                optimizer.step()

        def eval_after_training(model, test_dl, y_validation, validation_weight, params, show_plots=False):
            model.eval()
            with torch.no_grad():
                y_pred_list = []
                for i_batch, (xb_acc, xb_gyr, xb_vid, xb_others, yb, lens) in enumerate(test_dl):
                    y_pred_list.append(model(xb_acc, xb_gyr, xb_vid, xb_others, lens).detach().cpu())
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
            model.eval()
            def vali_set_loss(nn_model, validation_dl, loss_fn):
                validation_loss = []
                for xb_acc, xb_gyr, xb_vid, xb_others, yb, lens in validation_dl:
                    with torch.no_grad():
                        yb_pred = nn_model(xb_acc, xb_gyr, xb_vid, xb_others, lens)
                        validation_loss.append(loss_fn(yb_pred, yb).item() / xb_acc.shape[0])
                return np.mean(validation_loss)

            vali_from_train_loss = vali_set_loss(model, vali_from_train_dl, loss_fn)
            vali_from_test_loss = vali_set_loss(model, vali_from_test_dl, loss_fn)
            logging.info("\t{:3}\t{:15.2f}\t{:15.2f}\t{:13.2f}s\t\t".format(
                i_epoch, vali_from_train_loss, vali_from_test_loss, time.time() - epoch_end_time))

        def loss_fn(y_pred, yb):
            weights = torch.Tensor([1, 2]).cuda()
            pct_var = (y_pred - yb) ** 2
            out = pct_var * weights
            loss = out.sum()
            return loss

        self.train_step_lens, self.validation_step_lens = self._get_step_len(x_train), self._get_step_len(x_validation)
        model = self.model().cuda()
        model.set_scalars(self._data_scalar)

        hyper_param = {'epoch': globals()['epoch'], 'batch_size': globals()['batch_size'], 'lr': globals()['lr'],
                       'use_ratio': 100, 'target_name': 'main_output', 'fields': ['EXT_KM_X', 'EXT_KM_Y']}
        params = SimpleNamespace(**hyper_param)
        train_dl, vali_from_train_dl, vali_from_test_dl, test_dl = prepare_data(
            self.train_step_lens, self.validation_step_lens, int(params.batch_size))
        """ Phase I training """
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        logging.info('\tEpoch | Validation_set_Loss | Test_set_Loss | Duration\t\t')
        epoch_end_time = time.time()
        for i_epoch in range(params.epoch):
            eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch)
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, loss_fn, params)
        eval_after_training(model, test_dl, y_validation, validation_weight, params)

        """ Phase II training """
        params.lr = params.lr / 10
        params.batch_size = params.batch_size * 10
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        logging.info('\tEpoch | Validation_set_Loss | Test_set_Loss | Duration\t\t')
        epoch_end_time = time.time()
        for i_epoch in range(params.epoch):
            eval_during_training(model, vali_from_train_dl, vali_from_test_dl, loss_fn, epoch_end_time, i_epoch)
            epoch_end_time = time.time()
            train(model, train_dl, optimizer, loss_fn, params)
        eval_after_training(model, test_dl, y_validation, validation_weight, params)
        return {'model': model}

    def predict(self, model, x_test):
        nn_model = model['model']
        self.test_step_lens = self._get_step_len(x_test)
        x_acc = torch.from_numpy(x_test['input_acc']).float().cuda()
        x_gyr = torch.from_numpy(x_test['input_gyr']).float().cuda()
        x_vid = torch.from_numpy(x_test['input_vid']).float().cuda()
        x_others = np.concatenate([x_test['anthro'], x_test['high_level']], axis=2)
        x_others = torch.from_numpy(x_others).float().cuda()
        nn_model.eval()
        with torch.no_grad():
            test_ds = TensorDataset(x_acc, x_gyr, x_vid, x_others, torch.from_numpy(self.test_step_lens))
            test_dl = DataLoader(test_ds, batch_size=20)
            y_pred_list = []
            for i_batch, (xb_acc, xb_gyr, xb_vid, xb_others, lens) in enumerate(test_dl):
                y_pred_list.append(nn_model(xb_acc, xb_gyr, xb_vid, xb_others, lens).detach().cpu())
            y_pred = torch.cat(y_pred_list)
        y_pred = y_pred.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return {'main_output': y_pred}

    def save_model_and_results(self, test_sub_y, pred_sub_y, test_sub_weight, models, test_sub_name):
        save_path = os.path.join(self.result_dir, 'sub_models', test_sub_name)
        os.makedirs(save_path, exist_ok=True)
        for model_name, model in models.items():
            copied_model = copy.deepcopy(model)
            torch.save(copied_model.cpu(), os.path.join(save_path, model_name + '.pth'))

        results, columns = [], []
        for category, fields in self._y_fields.items():
            y_true_columns = fields
            columns += y_true_columns
            results.append(test_sub_y[category])
        for category, fields_data in pred_sub_y.items():
            if category == 'main_output':
                y_pred_columns = ['pred_' + field for field in self._y_fields['main_output']]
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

    @staticmethod
    def _append_stance_phase_feature(data, step_len):
        keys = list(set(list(data.keys())) - set(['anthro']))
        step_num = data[keys[0]].shape[0]
        max_len = data[keys[0]].shape[1]
        step_phase = np.zeros([step_num, max_len, 1])
        for i in range(0, step_num):
            step_phase[i, :step_len[i], 0] = np.linspace(0., 1., step_len[i])
        for k in keys:
            data[k] = np.concatenate([data[k], step_phase], 2)
        return data

    def hyperparam_tuning(self, hyper_train_sub_ids, hyper_vali_sub_ids):
        logging.info('Searching best hyper parameters, subjects for validation: {}'.format(hyper_vali_sub_ids))
        logging.disabled = True
        global hyper_train_fun, hyper_vali_fun, hyper_train_ids, hyper_vali_ids
        hyper_train_fun, hyper_vali_fun = self.preprocess_and_train, self.model_evaluation
        hyper_train_ids, hyper_vali_ids = hyper_train_sub_ids, hyper_vali_sub_ids
        best_param = {'epoch': 5, 'lr': 3e-3, 'batch_size': 20, 'lstm_unit': 40, 'fcnn_unit': 40}
        best_param = int_params(best_param)

        logging.disabled = False
        globals().update(best_param)
        best_param = {param: globals()[param] for param in ['epoch', 'lr', 'batch_size', 'lstm_unit', 'fcnn_unit']
                      if param in globals()}
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


def run(model, input_acc, input_gyr, input_vid, result_dir):
    x_fields = {'input_acc': input_acc, 'input_gyr': input_gyr, 'input_vid': input_vid}
    x_fields['anthro'] = STATIC_DATA
    x_fields['high_level'] = HIGH_LEVEL_FEATURE
    y_fields = {
        'main_output': ['EXT_KM_X', 'EXT_KM_Y'],
        'auxiliary_info': [SUBJECT_ID, TRIAL_ID, FORCE_PHASE]
    }
    weights = {key: [FORCE_PHASE] * len(y_fields[key]) for key in y_fields.keys()}
    evaluate_fields = {'main_output': y_fields['main_output']}

    model_builder = AlanFramework(data_path, model, x_fields, y_fields, TRIALS, weights, evaluate_fields, result_dir=result_dir)
    subjects = model_builder.get_all_subjects()
    model_builder.preprocess_train_evaluation(subjects[:-2], subjects[-2:], subjects[-2:])
    plt.close('all')

NO_INPUT = [FORCE_PHASE]
FEATURES_OTHERS = [WEIGHT, HEIGHT, FPA, TRUNK_SWAY, ANKLE_WIDTH] = range(5)
data_path = DATA_PATH + '/40samples+stance.h5'
VID_90_FIELDS = [loc + axis + '_90' for loc in USED_KEYPOINTS for axis in ['_x', '_y']]
VID_180_FIELDS = [loc + axis + '_180' for loc in USED_KEYPOINTS for axis in ['_x', '_y']]
VID_ALL = VID_90_FIELDS + VID_180_FIELDS

ACC_ALL = [field + '_' + sensor for sensor in SENSOR_LIST for field in IMU_FIELDS[:3]]
GYR_ALL = [field + '_' + sensor for sensor in SENSOR_LIST for field in IMU_FIELDS[3:6]]
ACC_3IMU = [field + '_' + sensor for sensor in ['L_FOOT', 'R_FOOT', 'WAIST'] for field in IMU_FIELDS[:3]]
GYR_3IMU = [field + '_' + sensor for sensor in ['L_FOOT', 'R_FOOT', 'WAIST'] for field in IMU_FIELDS[3:6]]
ACC_1IMU = [field + '_' + sensor for sensor in ['WAIST'] for field in IMU_FIELDS[:3]]
GYR_1IMU = [field + '_' + sensor for sensor in ['WAIST'] for field in IMU_FIELDS[3:6]]

if __name__ == "__main__":
    """ Use all the IMU channels """
    result_date = time.strftime('%y%m%d_%H%M')
    run(model=lambda: LmfImuOnlyNet(acc_dim=24, gyr_dim=24), input_acc=ACC_ALL, input_gyr=GYR_ALL, input_vid=NO_INPUT, result_dir=result_date + '/Lmf8Imu0Camera')
    run(model=lambda: LmfImuOnlyNet(acc_dim=9, gyr_dim=9), input_acc=ACC_3IMU, input_gyr=GYR_3IMU, input_vid=NO_INPUT, result_dir=result_date + '/Lmf3Imu0Camera')
    run(model=lambda: LmfImuOnlyNet(acc_dim=3, gyr_dim=3), input_acc=ACC_1IMU, input_gyr=GYR_1IMU, input_vid=NO_INPUT, result_dir=result_date + '/Lmf1Imu0Camera')

