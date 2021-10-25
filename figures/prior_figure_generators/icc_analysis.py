import rpy2
import os
import h5py
import json
from figures.PaperFigures import get_mean_std, format_axis
from const import SUBJECTS
import numpy as np
from const import LINE_WIDTH, FONT_DICT, FONT_SIZE, FONT_DICT, FONT_SIZE, FONT_DICT_LARGE
from const import LINE_WIDTH_THICK, FONT_SIZE_LARGE, FORCE_PHASE
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import pingouin as pg


def read_data_to_df(data_file):
    with h5py.File(data_file, 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    data_array = np.concatenate(list(data_all_sub.values()), axis=0).reshape([-1, len(data_fields)])
    data_df = pd.DataFrame(data_array, columns=data_fields)
    stance_data_df = data_df.loc[data_df[FORCE_PHASE] == 1].reset_index(drop=True)
    return stance_data_df, data_df


if __name__ == "__main__":
    stance_data_df, all_df = read_data_to_df('results/0114_KAM/IMU+OP/results.h5')

    plt.figure()
    plt.plot(stance_data_df['true_midout_r_z'])
    plt.plot(stance_data_df['pred_midout_r_z_pre'])
    plt.plot(stance_data_df['pred_midout_r_z'])
    plt.show()

    target_score = np.row_stack([stance_data_df[['subject_id', 'trial_id', 'true_main_output']].values, stance_data_df[['subject_id', 'trial_id', 'pred_main_output']].values])
    raters = np.concatenate([np.full([stance_data_df.shape[0]], 1), np.full([stance_data_df.shape[0]], 2)])
    step_index = np.concatenate([stance_data_df.index, stance_data_df.index])
    df = pd.DataFrame(np.column_stack([target_score, raters, step_index]), columns=["subject", "trial", "scores", "raters", "step_index"])

    print(pg.intraclass_corr(data=df, targets='subject', raters='raters', ratings='scores'))
    print(pg.intraclass_corr(data=df, targets='trial', raters='raters', ratings='scores'))
    print(pg.intraclass_corr(data=df, targets='step_index', raters='raters', ratings='scores'))

