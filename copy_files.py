import os
from const import SUBJECTS, STATIC_TRIALS, TRIALS
import shutil


dir_ori = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2021_VideoIMUCombined/experiment_data/KAM/'
dir_target = 'G:/Shared drives/NMBL Shared Data/datasets/Tan2022/Raw/'

for subject in SUBJECTS[:]:
    os.makedirs(os.path.join(dir_target, subject), exist_ok=True)
    for trial in STATIC_TRIALS+TRIALS:
        src = os.path.join(dir_ori, subject, 'vicon', trial + '.csv')
        dst = os.path.join(dir_target, subject, trial+'.csv')
        shutil.copyfile(src, dst)









