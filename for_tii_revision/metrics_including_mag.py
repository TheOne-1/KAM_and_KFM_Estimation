from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from const import GRAVITY


COMPARED_MODELS_FOR_PRINT = {
    'LmfMagNet': 'Our + Low-Rank Multimodal Fusion Network with Mag',
    'LmfNet': 'Ours + Low-Rank Multimodal Fusion Network', 'TfnNet': 'Ours + Tensor Fusion Network',
}
COMPARED_MODELS = list(COMPARED_MODELS_FOR_PRINT.keys())

if __name__ == '__main__':
    result_date = '../figures/results/1028'
    metric = 'RMSE'
    result_df = pd.read_csv(result_date + '/estimation_result_individual.csv')
    trial_df = result_df[result_df['trial'] == 'all']
    to_print = []
    for model in COMPARED_MODELS:
        one_row = ''
        one_row += '{:52}'.format(COMPARED_MODELS_FOR_PRINT[model])
        for moment_name in ['KAM', 'KFM']:
            sensor_result = trial_df[metric + '_' + model + '_' + moment_name]
            if model in ['TfnNet', 'LmfNet'] and metric == 'RMSE':
                sensor_result = sensor_result / GRAVITY * 100
            one_row = one_row + '&{:6.2f} ({:3.2f})'.format(np.mean(sensor_result), sensor_result.sem()) + '\t'
            if moment_name == 'KAM':
                one_row = one_row + '&\t'
        one_row = one_row + '\\\\'
        to_print.append(one_row)

    to_print.insert(4, '\cmidrule{1-8}')
    for row in to_print:
        print(row)

    for moment_name in ['KAM', 'KFM']:
        p_val = ttest_rel(trial_df[metric + '_LmfMagNet_' + moment_name].values,
                          trial_df[metric + '_LmfNet_' + moment_name].values / GRAVITY * 100).pvalue
        print(p_val)






