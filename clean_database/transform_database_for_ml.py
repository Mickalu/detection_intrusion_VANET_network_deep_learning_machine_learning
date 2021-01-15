# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:23:24 2021

@author: lucas
"""

import pandas as pd

PATH_DATA = "D:/School/cours_5eme/projet/database/Kyoto2016/2015/01/csv_file/"
DATA_FILE_NAME = "clean_20150101.csv"


df = pd.read_csv(PATH_DATA + DATA_FILE_NAME, sep = "\t")

columns_name = ['duration_connection','service','source_bytes','destination_bytes','count','same_srv_rate','serror_rate','srv_serror_rate','dst_host_count','dst_host_srv_count','dst_host_same_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','flag','IDS_detection','malware_detection','ashula_detection','label','source_IP_address','source_port_number','destination_IP_address','destination_port_number','start_time','duration']
df.columns = columns_name

df.to_csv('D:/School/cours_5eme/projet/database/Kyoto2016/2015/01/csv_file/ml_clean_20150101.csv')