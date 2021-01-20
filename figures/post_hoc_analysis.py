from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS


REMOVE_SUBJECT_MEAN = True

if __name__ == '__main__':
    metric = 'RMSE'
    result_names = ['_IMU_OP', '_IMU', '_OP']
    for target in ['KAM']:
        result_df = pd.read_csv('./exports/' + target + '_estimation_result_individual.csv')
        for trial in TRIALS + ['all']:
            trial_df = result_df[result_df['trial'] == trial][[metric + result_name for result_name in result_names]]
            if REMOVE_SUBJECT_MEAN:
                sub_mean = np.mean(trial_df.values, axis=1)
                trial_df = trial_df.subtract(sub_mean, axis=0)
            trial_df = trial_df.melt(var_name='groups', value_name='values')
            posthoc_result = posthoc_tukey(trial_df, val_col='values', group_col='groups')
            print('\n' + trial)
            print(posthoc_result)
