from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS, TRIALS_PRINT
from scipy.stats import ttest_1samp, ttest_rel


if __name__ == '__main__':
    result_date = 'results/0326'
    result_names = ['8IMU_2camera', '8IMU', '2camera']
    metric_incre = 'rRMSE_'

    for target in ['KAM', 'KFM']:
        result_df = pd.read_csv(result_date + target + '/estimation_result_individual.csv')

        """ print table """
        for trial, trial_print in zip(['baseline', 'fpa', 'step_width', 'trunk_sway'], ['Baseline', 'FPA', 'Step Width', 'Trunk Sway']):
            trial_df = result_df[result_df['trial'] == trial][[metric_incre + result_name for result_name in result_names]]
            print('\t\t& {:12}'.format(trial_print), end='')
            imu_camera_result = trial_df[metric_incre + result_names[0]]
            print('&{:6.1f} ({:3.1f})'.format(np.mean(imu_camera_result), imu_camera_result.sem()), end='\t')
            for result in result_names[1:]:
                sensor_result = trial_df[metric_incre + result]
                significance = ''
                if ttest_rel(imu_camera_result, sensor_result).pvalue < 0.05:
                    significance = '*'
                print('&{:6.1f} ({:3.1f}) {:1}'.format(np.mean(sensor_result), sensor_result.sem(), significance), end='\t')
            print('\\\\')



