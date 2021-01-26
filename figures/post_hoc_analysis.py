from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS, TRIALS_PRINT
from scipy.stats import ttest_1samp, ttest_ind


if __name__ == '__main__':
    result_date = 'results/0122_'
    result_names = ['_IMU_OP', '_IMU', '_OP']

    """ t test to compare whether the increase is significantly larger than zero """
    for target in ['KAM', 'KFM']:
        result_df = pd.read_csv(result_date + target + '/estimation_result_individual.csv')

        "print overall increase percent"
        metric_incre = 'RMSE'
        overall_df = result_df[result_df['trial'] == 'all']
        for result in result_names[1:]:
            increase_percent = (overall_df[metric_incre + result] - overall_df[metric_incre + '_IMU_OP']) / overall_df[metric_incre + '_IMU_OP'] * 100
            # print(target + '\t' + "%.1f ± %.1f, " % [np.mean(increase_percent), np.std(increase_percent)])
            print('{:10}{:10}{:6.1f} ± {:4.1f}'.format(target, result[1:], np.mean(increase_percent), np.std(increase_percent)))

        """ print table """
        for i_trial, trial in enumerate(TRIALS):
            trial_df = result_df[result_df['trial'] == trial][[metric_incre + result_name for result_name in result_names]]
            print('\t\t& {:12}'.format(TRIALS_PRINT[i_trial]), end='')
            for result in result_names[1:]:
                increase_percent = (trial_df[metric_incre + result] - trial_df[metric_incre + '_IMU_OP']) / trial_df[metric_incre + '_IMU_OP'] * 100
                significance = ''
                if ttest_1samp(increase_percent, 0.).pvalue < 0.05:
                    significance = '*'
                print('&{:6.1f} ({:4.1f}) {:1}'.format(np.mean(increase_percent), np.std(increase_percent), significance), end='\t')
            print('\\\\')

        "print overall increase percent"
        metric_feature = 'rRMSE'
        result_all_feature_df = pd.read_csv(result_date + 'used_all_the_features_' + target + '/estimation_result_individual.csv')
        overall_all_feature_df = result_all_feature_df[result_all_feature_df['trial'] == 'all']
        for result in result_names:
            result_all = overall_all_feature_df[metric_feature + result]
            result_sel = overall_df[metric_feature + result]
            ttest_res = ttest_ind(result_all, result_sel)
            print(ttest_res.pvalue)








