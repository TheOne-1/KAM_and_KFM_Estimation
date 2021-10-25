from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from const import COMPARED_MODELS_FOR_PRINT, SUBJECTS, COMPARED_MODELS
from scipy.stats import ttest_1samp, ttest_rel


if __name__ == '__main__':
    result_date = 'results/1018'
    metrics = ['RMSE', 'rRMSE', 'r']
    result_df = pd.read_csv(result_date + '/estimation_result_individual.csv')
    """ print table """
    to_print = []
    for model in COMPARED_MODELS:
        one_row = ''
        trial_df = result_df[result_df['trial'] == 'all']
        one_row += '{:22}'.format(COMPARED_MODELS_FOR_PRINT[model])
        for moment_name in ['KAM', 'KFM']:
            for metric in metrics:
                sensor_result = trial_df[metric + '_' + model + '_' + moment_name]
                if metric == 'rRMSE':
                    one_row = one_row + '&{:6.1f} ({:3.1f})'.format(np.mean(sensor_result), sensor_result.sem()) + '\t'
                else:
                    one_row = one_row + '&{:6.2f} ({:3.2f})'.format(np.mean(sensor_result), sensor_result.sem()) + '\t'
            if moment_name == 'KAM':
                one_row = one_row + '&\t'
        one_row = one_row + '\\\\'
        to_print.append(one_row)

    for start_index, sign in zip([100, 86, 71, 54, 40, 25], [-1, 1, 1, -1, 1, 1]):
        numbers = []
        for one_row in to_print:
            numbers.append(sign*float(one_row[start_index:start_index+4]))
            # print(one_row[start_index:start_index+4])
        bold_rows = np.where(numbers == np.min(numbers))[0]
        for bold_row in bold_rows:
            the_row = to_print[bold_row]
            if ' ' == the_row[start_index:start_index+4][0]:
                text_index, text_span = start_index + 1, 3
            else:
                text_index, text_span = start_index, 4
            to_print[bold_row] = the_row[: text_index] + '\\textbf{' + the_row[text_index:text_index + text_span] + \
                                 '}' + the_row[text_index + text_span:]

    to_print.insert(4, '\cmidrule{1-8}')
    for row in to_print:
        print(row)

