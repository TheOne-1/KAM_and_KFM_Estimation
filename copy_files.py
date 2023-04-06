import os
from const import SUBJECTS, STATIC_TRIALS, TRIALS
import shutil


dir_ori = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2021_VideoIMUCombined/experiment_data/KAM/'
dir_target = 'G:/Shared drives/NMBL Shared Data/datasets/Tan2022/Raw/'

for subject in SUBJECTS[:]:
    subject_id_short = subject[:4]
    os.makedirs(os.path.join(dir_target, subject_id_short), exist_ok=True)
    for trial in STATIC_TRIALS+TRIALS:
        src = os.path.join(dir_ori, subject, 'combined', trial + '.csv')
        dst = os.path.join(dir_target, subject_id_short, trial+'.csv')
        shutil.copyfile(src, dst)









