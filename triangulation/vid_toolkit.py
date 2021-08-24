import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sym
from numpy import cos, sin, arctan2, arcsin
import os
import glob
from const import TREADMILL_MAG_FIELD, DATA_PATH, GRAVITY, TARGETS_LIST
from types import SimpleNamespace
from transforms3d.quaternions import rotate_vector, mat2quat, quat2mat
from transforms3d.euler import mat2euler


def angle_between_vectors(subject, trial, segment, init_params):