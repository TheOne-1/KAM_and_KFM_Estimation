from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS, TRIALS_PRINT
from scipy.stats import ttest_1samp, ttest_rel


if __name__ == '__main__':
    result_date = 'results/1028'
    model_name = ['TfnNet', 'Lmf8Imu0Camera', 'Lmf0Imu2Camera']
    metric_incre = 'rRMSE_'
    result_df = pd.read_csv(result_date + '/estimation_result_individual.csv')

    for target in ['KAM', 'KFM']:
        for trial, trial_print in zip(['baseline', 'fpa', 'step_width', 'trunk_sway'], ['Baseline', 'Foot Progression Angle', 'Step Width', 'Trunk Sway']):
            trial_df = result_df[result_df['trial'] == trial]
            print('\t\t& {:12}'.format(trial_print), end='')
            imu_camera_result = trial_df[metric_incre + model_name[0] + '_' + target]
            print('&{:6.1f} ({:3.1f})'.format(np.mean(imu_camera_result), imu_camera_result.sem()), end='\t')
            for model in model_name[1:]:
                sensor_result = trial_df[metric_incre + model + '_' + target]
                significance = ''
                if ttest_rel(imu_camera_result, sensor_result).pvalue < 0.05:
                    significance = '*'
                print('&{:6.1f} ({:3.1f}) {:1}'.format(np.mean(sensor_result), sensor_result.sem(), significance), end='\t')
            print('\\\\')


