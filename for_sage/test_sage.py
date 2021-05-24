import numpy as np
from const import SUBJECTS, SENSOR_LIST, IMU_FIELDS
import pandas as pd
from a_load_model_and_predict import *


def create_test_input():
    with h5py.File('../trained_models_and_example_data/example_data_temp.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    sensor_col_name = [axis + '_' + sensor for sensor in SENSOR_LIST for axis in IMU_FIELDS[:6]]
    col_loc = [data_fields.index(col_name) for col_name in sensor_col_name]
    imu_data = data_all_sub['subject_01'][:, :, col_loc]
    imu_data[np.isnan(imu_data)] = 0.
    imu_data = imu_data.reshape([-1, imu_data.shape[2]])
    fill_loc = np.where(imu_data[:, 9] == 0.)
    imu_data[fill_loc, 9] = -300

    sensor_col_name_sage = [axis + '_' + str(i_sensor) for i_sensor in range(8) for axis in IMU_FIELDS[:6]]
    test_input_df = pd.read_csv('/media/tan/CPBA_X64FRE/knee_moments/test_input.csv', index_col=False)
    test_input_df.iloc[:imu_data.shape[0]][sensor_col_name_sage] = imu_data
    test_input_df.to_csv('/media/tan/CPBA_X64FRE/knee_moments/test_input.csv', index=False)


def get_body_weighted_imu(data_all_sub, data_fields):
    weight_col_loc = data_fields.index('body weight')
    for sub_name, sub_data in data_all_sub.items():
        sub_weight = sub_data[0, 0, weight_col_loc]
        for segment in ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']:
            segment_imu_col_loc = [data_fields.index(field + '_' + segment) for field in IMU_FIELDS[:6]]
            sub_data[:, :, segment_imu_col_loc[:3]] = \
                sub_data[:, :, segment_imu_col_loc[:3]] * sub_weight * SEGMENT_MASS_PERCENT[segment] / 100
        data_all_sub[sub_name] = sub_data


def compare_real_time_and_offline_results():
    """ step 0: select model and load data """
    # Five models are available: 8IMU_camera, 3IMU_camera, 8IMU, 3IMU, camera
    model_name = '8IMU'
    # Two target moments: KAM or KFM
    target_moment = 'KAM'

    assert model_name in ['8IMU_camera', '3IMU_camera', '8IMU', '3IMU', 'camera'], 'Incorrect model name.'
    assert target_moment in ['KAM', 'KFM'], 'Incorrect target moment name.'

    # one example data file is available
    with h5py.File('../trained_models_and_example_data/example_data_temp.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    model_path = './sage_model/' + model_name + '_' + target_moment + '.pth'
    model = torch.load(model_path)

    """ step 1: prepare subject 01's data as input """
    get_body_weighted_imu(data_all_sub, data_fields)

    subject_data = data_all_sub['subject_01']
    model_inputs = {}
    model_inputs['anthro'] = torch.from_numpy(subject_data[:, :, [data_fields.index('body weight'),
                                                                  data_fields.index('body height')]])
    model_inputs['step_length'] = torch.from_numpy(np.sum(~(subject_data[:, :, 0] == 0.), axis=1))

    for submodel, component in zip([model.model_fx, model.model_fz, model.model_rx, model.model_rz],
                                   ['force_x', 'force_z', 'r_x', 'r_z']):
        input_fields_ = submodel.input_fields
        data_to_process = copy.deepcopy(subject_data)

        other_feature_loc = [data_fields.index(field) for field in input_fields_ if 'Acc' not in field]
        data_to_process[:, :, other_feature_loc] = normalize_array_separately(
            data_to_process[:, :, other_feature_loc], model.scalars[component + '_other'], 'transform', scalar_mode='by_each_column')

        weighted_acc_loc = [data_fields.index(field) for field in input_fields_ if 'Acc' in field]
        if len(weighted_acc_loc) > 0:
            data_to_process[:, :, weighted_acc_loc] = normalize_array_separately(
                data_to_process[:, :, weighted_acc_loc], model.scalars[component + '_acc'], 'transform', scalar_mode='by_all_columns')
        submodel_input = data_to_process[:, :, [data_fields.index(field) for field in input_fields_]]
        model_inputs[component] = torch.from_numpy(submodel_input)

    """ step 2: predict moment of subject 01 """
    predicted = model(model_inputs['force_x'], model_inputs['force_z'], model_inputs['r_x'], model_inputs['r_z'],
                      model_inputs['anthro'], model_inputs['step_length']).detach().numpy()
    if target_moment == 'KFM':
        predicted = - predicted

    """ step 3: plot estimation and true values """
    if target_moment == 'KAM':
        ground_truth_moment = subject_data[:, :, data_fields.index('EXT_KM_Y')]
    else:
        ground_truth_moment = -subject_data[:, :, data_fields.index('EXT_KM_X')]
    # change unit of body weight from Kg to N
    # ground_truth_moment, predicted = GRAVITY * ground_truth_moment, GRAVITY * predicted
    plt.figure()
    new_data = pd.read_csv('trial_0.csv', index_col=False)
    new_kam = new_data['KAM']
    plt.plot(new_kam, label='Sage Value')
    plt.plot(predicted.ravel(), label='PC Value')
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel('Time Step')
    ax.set_ylabel(target_moment + ' (BW X BH)')
    plt.title(model_name + ' model')
    plt.show()

if __name__ == "__main__":

    # create_test_input()

    compare_real_time_and_offline_results()



