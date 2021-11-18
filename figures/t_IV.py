import pandas as pd
import numpy as np
from scipy.stats import sem
from const import SENSOR_LIST


def print_t_IV_report_increase_of_RMSE():
    results_df = pd.read_csv('../for_tii_revision/placement_error_results.csv')
    metric_name = 'RMSE_'
    """ No error """

    config_single_pos = {'start_str': '\multirow{2}{*}{One of Eight}\t& Position & 100 mm',
                         'error_to_loop': error_names[:1],
                         'sensor_to_loop': SENSOR_LIST}
    config_single_ori = {'start_str': '& Orientation & 10 deg',
                         'error_to_loop': error_names[2:],
                         'sensor_to_loop': SENSOR_LIST}
    config_multiple_pos = {'start_str': '\multirow{2}{*}{All Eight}\t& Position & 100 mm',
                           'error_to_loop': error_names[:1],
                           'sensor_to_loop': ['all']}
    config_multiple_ori = {'start_str': '& Orientation & 10 deg',
                           'error_to_loop': error_names[2:],
                           'sensor_to_loop': ['all']}

    for i_config, config in enumerate([config_single_pos, config_single_ori, config_multiple_pos, config_multiple_ori]):
        print(config['start_str'], end='\t')
        for i_model, model_name in enumerate(model_names):
            for target in ['KAM', 'KFM']:
                mean_, sem_ = 0, 0
                for error_name in config['error_to_loop']:
                    for sensor in config['sensor_to_loop']:
                        no_error_df = results_df[(results_df['trial'] == 'all') & (results_df['error_type'] == 'no') & (results_df['model_name'] == model_name)]
                        condition_df = results_df[(results_df['trial'] == 'all') & (results_df['error_segment'] == sensor) &
                                                  (results_df['error_name'] == error_name) & (results_df['model_name'] == model_name)]
                        increases_ = condition_df[metric_name + target].values - no_error_df[metric_name + target].values
                        if np.mean(increases_) > mean_:
                            mean_, sem_ = np.mean(increases_), sem(increases_)
                print('&{:6.2f} ({:3.2f})'.format(mean_, sem_), end='\t')
            if i_model == 0:
                print('& ', end='')
        print('\\\\')
        if i_config == 1:
            print('\cmidrule{1-8}')


error_names = ['_e_pos_x', '_e_pos_z', '_e_ori_z']
model_names = ['TfnNet', 'Lmf8Imu0Camera']

if __name__ == "__main__":
    print_t_IV_report_increase_of_RMSE()


