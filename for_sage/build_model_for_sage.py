from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from alan_framework import AlanFramework, combine_imu_vid_fields
from alan_framework import input_imu_8_all, input_imu_3_all, input_imu_1_all
from const import FORCE_PHASE, DATA_PATH, SEGMENT_MASS_PERCENT, TRIALS, SUBJECT_ID, TRIAL_ID, STATIC_DATA
import copy
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class FourSourceModel(nn.Module):
    def __init__(self, model_fx, model_fz, model_rx, model_rz, scalars):
        super(FourSourceModel, self).__init__()
        self.model_fx = model_fx
        self.model_fz = model_fz
        self.model_rx = model_rx
        self.model_rz = model_rz
        self.scalars = scalars

    def forward(self, x_fx, x_fz, x_rx, x_rz, anthro, lens):
        out_fx = self.model_fx(x_fx, lens)
        out_fz = self.model_fz(x_fz, lens)
        out_rx = self.model_rx(x_rx, lens)
        out_rz = self.model_rz(x_rz, lens)
        zero_padding_loc = (out_fx == 0.) & (out_fz == 0.) & (out_rx == 0.) & (out_rz == 0.)
        out_fx = self.inverse_scaling(out_fx, 'midout_force_x')
        out_fz = self.inverse_scaling(out_fz, 'midout_force_z')
        out_rx = self.inverse_scaling(out_rx, 'midout_r_x')
        out_rz = self.inverse_scaling(out_rz, 'midout_r_z')
        weight = anthro[:, 0, 0].unsqueeze(1).unsqueeze(2)
        height = anthro[:, 0, 1].unsqueeze(1).unsqueeze(2)
        output = out_fx * out_rz - out_fz * out_rx
        output = torch.div(output, weight * height)
        output[zero_padding_loc] = 0
        return output

    def inverse_scaling(self, data, fields):
        data[data == 0.] = np.nan
        if isinstance(self.scalars[fields], MinMaxScaler):
            bias_, scale_ = self.scalars[fields].min_[0], self.scalars[fields].scale_[0]
        elif isinstance(self.scalars[fields], StandardScaler):
            bias_, scale_ = self.scalars[fields].mean_[0], self.scalars[fields].scale_[0]
        data = torch.add(data, - bias_)
        data = torch.div(data, scale_)
        data[torch.isnan(data)] = 0.
        return data


class TianRNN(nn.Module):
    def __init__(self, x_dim, y_dim, input_fields, seed=0, nlayer=2):
        super(TianRNN, self).__init__()
        torch.manual_seed(seed)
        self.rnn_layer = nn.LSTM(x_dim, globals()['lstm_unit'], nlayer, batch_first=True, bidirectional=True)
        self.y_dim = y_dim
        self.r2d = nn.Linear(2 * globals()['lstm_unit'], globals()['fcnn_unit'], bias=False)
        self.d2o = nn.Linear(globals()['fcnn_unit'], y_dim, bias=False)
        self.relu = nn.ReLU()
        self.input_fields = input_fields
        for layer in [self.r2d, self.d2o]:
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
        sequence = self.r2d(lstm_out)
        sequence = self.relu(sequence)
        output = self.d2o(sequence)
        return output


class AlanFrameworkSage(AlanFramework):
    def __init__(self,  *args, **kwargs):
        self.model_name = kwargs.pop('model_name')
        AlanFramework.__init__(self, *args, **kwargs)

    def train_model(self, x_train, y_train, x_validation=None, y_validation=None, validation_weight=None):
        sub_model_hyper_param = {'epoch': globals()['epoch_1'], 'batch_size': globals()['batch_size_1'],
                                 'lr': globals()['lr_1'], 'weight_decay': 0, 'use_ratio': 100}
        self.train_step_lens, self.validation_step_lens = self._get_step_len(x_train), self._get_step_len(x_validation)

        sub_models = []
        for i_target, target, field in zip([0, 1, 2, 3], ['force_x', 'force_z', 'r_x', 'r_z'],
                                           ['plate_2_force_x', 'plate_2_force_z', 'r_x', 'r_z']):
            x_train_sub, x_validation_sub = x_train[target], x_validation[target]
            y_train_sub, y_validation_sub = y_train['midout_'+target], y_validation['midout_'+target]
            model_sub = TianRNN(x_train_sub.shape[2], y_train_sub.shape[2], self._x_fields[target], i_target).cuda()
            params = {**sub_model_hyper_param, **{'target_name': 'midout_'+target, 'fields': [field]}}
            self.build_sub_model(model_sub, x_train_sub, y_train_sub, x_validation_sub, y_validation_sub, validation_weight, params)
            sub_models.append(model_sub)

        model_fx, model_fz, model_rx, model_rz = sub_models
        four_source_model = FourSourceModel(model_fx, model_fz, model_rx, model_rz, self._data_scalar)
        four_source_model_pre = copy.deepcopy(four_source_model).cuda()

        main_model_hyper_param = {'epoch': globals()['epoch_1'], 'batch_size': globals()['batch_size_1']*10,
                                  'lr': globals()['lr_1']/10, 'weight_decay': 0,
                                  'use_ratio': sub_model_hyper_param['use_ratio']}
        main_model_hyper_param.update({'target_name': 'main_output', 'fields': ['EXT_KM_Y']})
        self.build_main_model(four_source_model, x_train, y_train, x_validation, y_validation, validation_weight, main_model_hyper_param)
        built_models = {'four_source_model': four_source_model, 'four_source_model_pre': four_source_model_pre,
                        'model_rx': model_rx, 'model_rz': model_rz, 'model_fx': model_fx, 'model_fz': model_fz}
        return built_models

    def save_model_and_results(self, test_sub_y, pred_sub_y, test_sub_weight, models, test_sub_name):
        os.makedirs(self.result_dir, exist_ok=True)
        save_path = os.path.join(self.result_dir, self.model_name + '.pth')
        model = copy.deepcopy(models['four_source_model']).cpu()
        torch.save(model, save_path)

    def predict(self, model, x_test):
        nn_model, four_source_model_pre = model['four_source_model'], model['four_source_model_pre']
        model_fx, model_fz, model_rx, model_rz = model['model_fx'], model['model_fz'], model['model_rx'], model['model_rz']
        self.test_step_lens = self._get_step_len(x_test)
        x_fx, x_fz, x_rx, x_rz, x_anthro = x_test['force_x'], x_test['force_z'], x_test['r_x'], x_test['r_z'], x_test[
            'anthro']
        x_fx, x_fz = torch.from_numpy(x_fx).float().cuda(), torch.from_numpy(x_fz).float().cuda()
        x_rx, x_rz, x_anthro = torch.from_numpy(x_rx).float().cuda(), torch.from_numpy(
            x_rz).float().cuda(), torch.from_numpy(x_anthro).float().cuda(),
        with torch.no_grad():
            test_ds = TensorDataset(x_fx, x_fz, x_rx, x_rz, x_anthro, torch.from_numpy(self.test_step_lens))
            test_dl = DataLoader(test_ds, batch_size=20)
            y_pred_list, y_pred_list_pre = [], []
            y_fx, y_fz, y_rx, y_rz = [], [], [], []
            for i_batch, (xb_fx, xb_fz, xb_rx, xb_rz, xb_anthro, lens) in enumerate(test_dl):
                y_pred_list.append(nn_model(xb_fx, xb_fz, xb_rx, xb_rz, xb_anthro, lens).detach().cpu())
                y_pred_list_pre.append(four_source_model_pre(xb_fx, xb_fz, xb_rx, xb_rz, xb_anthro, lens).detach().cpu())
                y_fx.append(model_fx(xb_fx, lens).detach().cpu())
                y_fz.append(model_fz(xb_fz, lens).detach().cpu())
                y_rx.append(model_rx(xb_rx, lens).detach().cpu())
                y_rz.append(model_rz(xb_rz, lens).detach().cpu())
            self.typical_input = (xb_fx.cpu(), xb_fz.cpu(), xb_rx.cpu(), xb_rz.cpu(), xb_anthro.cpu(), lens.cpu())
            y_pred, y_pred_pre = torch.cat(y_pred_list), torch.cat(y_pred_list_pre)
            y_fx, y_fz, y_rx, y_rz = torch.cat(y_fx), torch.cat(y_fz), torch.cat(y_rx), torch.cat(y_rz)
        y_pred = y_pred.detach().cpu().numpy()
        return {'main_output': y_pred, 'main_output_pre': y_pred_pre,
                'midout_force_x': y_fx, 'midout_force_z': y_fz, 'midout_r_x': y_rx, 'midout_r_z': y_rz}


def run_kam(input_imu, input_vid, result_dir, model_name):
    x_fields = {'force_x': [], 'force_z': [], 'r_x': [], 'r_z': []}
    x_fields = combine_imu_vid_fields(x_fields, input_imu, input_vid)
    y_fields = {
        'main_output': ['EXT_KM_Y'],
        'midout_force_x': ['plate_2_force_x'],
        'midout_force_z': ['plate_2_force_z'],
        'midout_r_x': ['r_x'],
        'midout_r_z': ['r_z'],
        'auxiliary_info': [SUBJECT_ID, TRIAL_ID, FORCE_PHASE]
    }
    run(x_fields, y_fields, ['EXT_KM_Y'], result_dir, model_name)


def run_kfm(input_imu, input_vid, result_dir, model_name):
    x_fields = {'force_y': [], 'force_z': [], 'r_y': [], 'r_z': []}
    x_fields = combine_imu_vid_fields(x_fields, input_imu, input_vid)
    y_fields = {
        'main_output': ['EXT_KM_X'],
        'midout_force_y': ['plate_2_force_y'],
        'midout_force_z': ['plate_2_force_z'],
        'midout_r_y': ['r_y'],
        'midout_r_z': ['r_z'],
        'auxiliary_info': [SUBJECT_ID, TRIAL_ID, FORCE_PHASE]
    }
    """ KFM can share the same modeling code by switching the axes: z -> y, x -> z"""
    x_fields_renamed = {'force_x': x_fields['force_z'], 'force_z': x_fields['force_y'],
                        'r_x': x_fields['r_z'], 'r_z': x_fields['r_y'], 'anthro': STATIC_DATA}
    y_fields_renamed = {'midout_force_x': y_fields['midout_force_z'], 'midout_force_z': y_fields['midout_force_y'],
                        'midout_r_x': y_fields['midout_r_z'], 'midout_r_z': y_fields['midout_r_y'],
                        'main_output': y_fields['main_output'], 'auxiliary_info': y_fields['auxiliary_info']}
    run(x_fields_renamed, y_fields_renamed, ['EXT_KM_X'], result_dir, model_name)


def run(x_fields, y_fields, main_output_fields, result_dir, model_name):
    weights = {key: [FORCE_PHASE] * len(y_fields[key]) for key in y_fields.keys()}
    weights.update({key: [FORCE_PHASE] * len(x_fields[key]) for key in x_fields.keys()})
    evaluate_fields = {'main_output': main_output_fields}

    model_builder = AlanFrameworkSage(data_path, x_fields, y_fields, TRIALS[:], weights, evaluate_fields,
                                      lambda: MinMaxScaler(feature_range=(-3, 3)), model_name=model_name, result_dir=result_dir)
    subjects = model_builder.get_all_subjects()
    model_builder.preprocess_train_evaluation(subjects[1:], subjects[:1], subjects[:1])
    plt.close('all')


def load_and_test(date, model_name):
    model_path = os.path.join(DATA_PATH, 'training_results', date, model_name+'.pth')
    model = torch.load(model_path)


if __name__ == "__main__":
    """ Train a smaller model """
    model_param = {'epoch_1': 5, 'lr_1': 3e-3, 'batch_size_1': 20, 'lstm_unit': 10, 'fcnn_unit': 15}
    globals().update(model_param)

    data_path = DATA_PATH + '/40samples+stance_StrikeOffNoFiltering.h5'
    result_date = '0520'
    run_kam(input_imu=input_imu_8_all, input_vid={}, result_dir=result_date, model_name='8IMU_KAM')
    run_kfm(input_imu=input_imu_8_all, input_vid={}, result_dir=result_date, model_name='8IMU_KFM')
    run_kam(input_imu=input_imu_3_all, input_vid={}, result_dir=result_date, model_name='3IMU_KAM')
    run_kfm(input_imu=input_imu_3_all, input_vid={}, result_dir=result_date, model_name='3IMU_KFM')
    run_kam(input_imu=input_imu_1_all, input_vid={}, result_dir=result_date, model_name='1IMU_KAM')
    run_kfm(input_imu=input_imu_1_all, input_vid={}, result_dir=result_date, model_name='1IMU_KFM')

    # load_and_test(result_date, '8IMU_KAM')

