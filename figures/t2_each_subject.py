"""Creating a figure for the thesis"""
import pandas as pd
from scipy.stats import pearsonr


if __name__ == '__main__':
    result_date = 'results/0326'
    result_name = '8IMU_2camera'
    metric = 'rRMSE_'

    results = []
    for target in ['KAM', 'KFM']:
        result_df = pd.read_csv(result_date + target + '/estimation_result_individual.csv')
        trial_df = result_df[result_df['trial'] == 'all'][[metric + result_name]]
        results.append(trial_df[metric + result_name].values)

    for i_sub in range(17):
        print('{:3.1f} \t\t {:3.1f}'.format(results[0][i_sub], results[1][i_sub]))

    print(pearsonr(results[0], results[1]))


