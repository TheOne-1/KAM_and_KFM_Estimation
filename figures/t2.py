from figures.PaperFigures import get_mean_std, get_data, get_fpa
from const import SUBJECTS
import numpy as np
from const import TRIALS


def get_moment_impulse(moment_data):
    


if __name__ == "__main__":
    for i_moment, moment in enumerate(['KAM', 'KFM']):
        """ Get baseline"""
        _data_all_bl, _data_fields = get_data('results/0131_all_feature_' + moment + '/8IMU_2camera/results.h5')
        _data_all_input, _data_fields_input = get_data('results/40samples+stance.h5')
        for i_sub in range(len(SUBJECTS)):
            _fpa_bl = get_fpa(_data_all_bl[i_sub], _data_fields)

