from scikit_posthocs import posthoc_tukey, posthoc_ttest
import pandas as pd
import numpy as np
from const import TRIALS_PRINT, SUBJECTS, COMPARED_MODELS, TRIALS
from sklearn.metrics import mean_squared_error as mse
from figures.PaperFigures import get_peak_of_each_gait_cycle
import h5py
import json
# from figures.f9 import sort_gait_cycles_according_to_param


if __name__ == '__main__':
    result_date = 'results/1028/'
    overall_result_df = pd.read_csv(result_date + 'estimation_result_individual.csv')

    models = ['LmfNet', 'Lmf8Imu0Camera', 'Lmf0Imu2Camera']        # DirectNet LmfNet
    data_all, data_fields_all = {}, {}
    for model in models:
        with h5py.File('{}{}/results.h5'.format(result_date, model), 'r') as hf:
            data_all[model] = {subject: subject_data[:] for subject, subject_data in hf.items()}
            data_fields_all[model] = json.loads(hf.attrs['columns'])

    to_print = ''
    for moment_name in ['KAM', 'KFM']:
        to_print += '\multirow{4}{*}{' + moment_name + '}'
        for trial, trial_to_print in zip(TRIALS, TRIALS_PRINT):
            to_print += ' & ' + trial_to_print + '\t'
            for model in models:
                data_subs, data_fields = data_all[model], data_fields_all[model]
                trial_df = overall_result_df[overall_result_df['trial'] == trial]
                overall_rmses = trial_df['RMSE_' + model + '_' + moment_name]
                to_print += '&{:6.2f} ({:3.2f})'.format(np.mean(overall_rmses), overall_rmses.sem()) + '\t'

                # peak results
                peak_rmses = []
                for i_subject, subject in enumerate(SUBJECTS):
                    sub_data = data_subs[subject]
                    ts_id = TRIALS.index(trial)
                    sub_trial_data = sub_data[sub_data[:, 0, data_fields.index('trial_id')] == ts_id, :, :]
                    true_peak, pred_peak = get_peak_of_each_gait_cycle(np.stack(sub_trial_data), data_fields, moment_name, 0.5)
                    peak_rmses.append(np.sqrt(mse(true_peak, pred_peak)))
                peak_rmses = pd.Series(peak_rmses)
                to_print += '&{:6.2f} ({:3.2f})'.format(np.mean(peak_rmses), peak_rmses.sem()) + '\t&\t'
            to_print = to_print[:-3]
            to_print += '\\\\\n'
        if moment_name == 'KAM':
            to_print += '\cmidrule{1-10}\n'
    print(to_print)

