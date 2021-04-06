import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder

list_day = ["01","02","03","04","05","06","07","08","09","10","11","12"]

for nbr_day in list_day:

    PATH_DIR_DATA = "D:/School/cours_5eme/projet/code/database/Kyoto2016/2015/"+ nbr_day +"/csv_file/"

    data_folder_csv = [f for f in listdir(PATH_DIR_DATA) if isfile(join(PATH_DIR_DATA, f))]

    for file in data_folder_csv:
    
        df_clean = pd.read_csv(PATH_DIR_DATA + file, sep = "\t")
        
        columns_name = ['duration_connection','service','source_bytes','destination_bytes','count','same_srv_rate','serror_rate','srv_serror_rate','dst_host_count','dst_host_srv_count','dst_host_same_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','flag','IDS_detection','malware_detection','ashula_detection','label','source_IP_address','source_port_number','destination_IP_address','destination_port_number','start_time','duration']
        list_encode_columns = ["service", "flag", "malware_detection", "ashula_detection", "duration", "IDS_detection"]
        list_str_columns = ['source_IP_address', 'destination_IP_address']
        
        
        list_str_encod = ['malware_detection','ashula_detection','duration','IDS_detection']

        df_clean.columns = columns_name
        
        df_clean = df_clean.drop(['start_time'], axis = 1)
        
        #on s'assure que les colonnes qui doivent être en string le sont bien
        for string_col in list_str_encod:
            df_clean[string_col] = df_clean[string_col].astype(str)
        
        #on encode les colonnes qui en ont besoin
        le = LabelEncoder()
        
        for col in list_encode_columns:
            df_clean[col] = le.fit_transform(df_clean[col])
        
        #encodage des adresses IP
        for column in list_str_columns:
            df_clean[column] = df_clean[column].astype('category')
            df_clean[column] = df_clean[column].cat.codes

        df_clean.to_csv( "D:/School/cours_5eme/projet/code/database/Kyoto2016/2015/"+ nbr_day +"/csv_file_clean/2_" + file)
            
        
    
