# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:48:42 2021

@author: lucas
"""

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join



PATH_DIR_DATA = "D:/School/cours_5eme/projet/database/Kyoto2016/2015/01/csv_file/"

df = pd.read_csv(PATH_DIR_DATA + "clean_20150120.csv", sep = "\t")
