import numpy as np
from const import SUBJECTS, SENSOR_LIST, IMU_FIELDS
import pandas as pd
import json
import h5py
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # with h5py.File('trained_models_and_example_data/example_data.h5', 'r') as hf:
    #     data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
    #     data_fields = json.loads(hf.attrs['columns'])
    #
    # sensor_col_name = [axis + '_' + sensor for sensor in SENSOR_LIST for axis in IMU_FIELDS[:6]]
    # col_loc = [data_fields.index(col_name) for col_name in sensor_col_name]
    # imu_data = data_all_sub['subject_01'][:, :, col_loc]
    # imu_data[np.isnan(imu_data)] = 0.
    # imu_data = imu_data.reshape([-1, imu_data.shape[2]])
    #
    # sensor_col_name_sage = [axis + '_' + str(i_sensor) for i_sensor in range(8) for axis in IMU_FIELDS[:6]]
    # test_input_df = pd.read_csv('test_input.csv', index_col=False)
    # test_input_df.iloc[:imu_data.shape[0]][sensor_col_name_sage] = imu_data
    # test_input_df.to_csv('test_input.csv', index=False)

    new_data = pd.read_csv('trial_0.csv', index_col=False)
    new_kam = new_data['KAM']

    with h5py.File('trained_models_and_example_data/example_data.h5', 'r') as hf:
        _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        _data_fields = json.loads(hf.attrs['columns'])

    old_data = _data_all_sub['subject_01']

    ground_truth_moment = old_data[:, :, _data_fields.index('EXT_KM_Y')]
    # plt.plot(ground_truth_moment.ravel())
    plt.plot(new_kam[new_kam != 0.])
    plt.show()


    # # change unit of body weight from Kg to N
    # ground_truth_moment, predicted = GRAVITY * ground_truth_moment, GRAVITY * predicted
    # plt.figure()
    # # plt.plot(ground_truth_moment.ravel(), label='True Value')
    # plt.plot(predicted.ravel(), label='Predicted Value')
    # # plt.legend()
    # # ax = plt.gca()
    # # ax.set_xlabel('Time Step')
    # # ax.set_ylabel(target_moment + ' (BW X BH)')
    # # plt.title(model_name + ' model')
    # # plt.show()
    #
    # import pandas as pd
    # new_data = pd.read_csv('trial_0.csv', index_col=False)
    # new_kam = new_data['KAM']
    #
    # with h5py.File('trained_models_and_example_data/example_data.h5', 'r') as hf:
    #     _data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
    #     _data_fields = json.loads(hf.attrs['columns'])
    #
    # old_data = _data_all_sub['subject_01']
    #
    # plt.plot(GRAVITY * new_kam[new_kam != 0.])
    # plt.show()





