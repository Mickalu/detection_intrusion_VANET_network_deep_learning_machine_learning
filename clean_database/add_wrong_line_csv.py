import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder

list_day = ["01", "02"]
list_line_malware_detected = []
columns_name = ['duration_connection','service','source_bytes','destination_bytes','count','same_srv_rate','serror_rate','srv_serror_rate','dst_host_count','dst_host_srv_count','dst_host_same_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','flag','IDS_detection','malware_detection','ashula_detection','label','source_IP_address','source_port_number','destination_IP_address','destination_port_number','start_time','duration']

for nbr_day in list_day:

    PATH_DIR_DATA = "D:/School/cours_5eme/projet/database/Kyoto2016/2015/"+ nbr_day +"/csv_file_clean/"

    data_folder_csv = [f for f in listdir(PATH_DIR_DATA) if isfile(join(PATH_DIR_DATA, f))]

    for file in data_folder_csv:
        df = pd.read_csv(PATH_DIR_DATA + file)

        malware_connection = df[df["malware_detection"] != 0]

        dataframe = pd.DataFrame(malware_connection)
        dataframe.to_csv("D:/School/cours_5eme/projet/database/Kyoto2016/2015/malware_detection/csv/"+file )
        

    


