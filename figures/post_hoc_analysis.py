from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS, TRIALS_PRINT
from scipy.stats import ttest_1samp


if __name__ == '__main__':
    result_date = 'results/0122_'       # _used_all_the_features_
    metric = 'RMSE'
    result_names = ['_IMU_OP', '_IMU', '_OP']

    """ t test to compare whether the increase is significantly larger than zero """
    for target in ['KAM', 'KFM']:
        result_df = pd.read_csv(result_date + target + '/estimation_result_individual.csv')
        for i_trial, trial in enumerate(TRIALS):
            trial_df = result_df[result_df['trial'] == trial][[metric + result_name for result_name in result_names]]
            print('\t\t& {:12}'.format(TRIALS_PRINT[i_trial]), end='')
            for result in result_names[1:]:
                increase_percent = (trial_df[metric + result] - trial_df[metric + '_IMU_OP']) / trial_df[metric + '_IMU_OP'] * 100
                significance = ''
                if ttest_1samp(increase_percent, 0.).pvalue < 0.05:
                    significance = '*'
                print('&{:6.1f} ({:4.1f}) {:1}'.format(np.mean(increase_percent), np.std(increase_percent), significance), end='\t')
            print('\\\\')

            # only_imu_increase = (trial_df[metric + '_IMU'] - trial_df[metric + '_IMU_OP']) / trial_df[metric + '_IMU_OP'] * 100
            # only_cam_increase = (trial_df[metric + '_OP'] - trial_df[metric + '_IMU_OP']) / trial_df[metric + '_IMU_OP'] * 100
            # p_value, _ = ttest_1samp(only_imu_increase, 0.)
            # print('\t\t& {:12}& {:6.1f} ({:4.1f}) &{:6.1f} ({:4.1f}) \\\\'.format(TRIALS_PRINT[i_trial],
            #     np.mean(only_imu_increase), np.std(only_imu_increase), np.mean(only_cam_increase), np.std(only_cam_increase)))



    # """ post hoc to compare actual values """
    # REMOVE_SUBJECT_MEAN = True
    # for target in ['KAM']:
    #     result_df = pd.read_csv(result_date + target + '/estimation_result_individual.csv')
    #     for trial in TRIALS + ['all']:
    #         trial_df = result_df[result_df['trial'] == trial][[metric + result_name for result_name in result_names]]
    #         if REMOVE_SUBJECT_MEAN:
    #             sub_mean = np.mean(trial_df.values, axis=1)
    #             trial_df = trial_df.subtract(sub_mean, axis=0)
    #         trial_df = trial_df.melt(var_name='groups', value_name='values')
    #         posthoc_result = posthoc_tukey(trial_df, val_col='values', group_col='groups')
    #         print('\n' + trial)
    #         print(posthoc_result)
