from scikit_posthocs import posthoc_tukey
import pandas as pd
import numpy as np
from const import TRIALS, SUBJECTS


if __name__ == '__main__':
    metric = 'rRMSE'
    result_names = ['_IMU_OP', '_IMU', '_OP']
    for target in ['KAM', 'KFM']:
        result_df = pd.read_csv('./exports/' + target + '_estimation_result_individual.csv')
        for trial in TRIALS + ['all']:
            trial_df = result_df[result_df['trial'] == trial][[metric + result_name for result_name in result_names]]
            trial_df = trial_df.melt(var_name='groups', value_name='values')
            posthoc_result = posthoc_tukey(trial_df, val_col='values', group_col='groups')
            print('\n' + trial)
            print(posthoc_result)
