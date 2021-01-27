import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelEncoder

list_day = ["09","10","11","12"]
list_df_wrong = []
list_df_not_wrong = []

for nbr_day in list_day:

    PATH_DIR_DATA = "D:/School/cours_5eme/projet/code/database/Kyoto2016/2015/"+ nbr_day +"/csv_file_clean/"

    data_folder_csv = [f for f in listdir(PATH_DIR_DATA) if isfile(join(PATH_DIR_DATA, f))]

    for file in data_folder_csv:
        df = pd.read_csv(PATH_DIR_DATA + file, error_bad_lines=False)
        print(file)
        malware_connection = df[df["malware_detection"] != 0]

        lenght_df_malware = malware_connection.shape[0]

        malware_not_detected = df[df["malware_detection"] == 0]
        malware_not_detected = malware_not_detected.iloc[:lenght_df_malware]
        
        

        list_df_wrong.append(malware_connection)
        list_df_not_wrong.append(malware_not_detected)
      


Df_result = pd.concat(list_df_wrong + list_df_not_wrong)


Df_result.to_csv("D:/School/cours_5eme/projet/code/database/Kyoto2016/2015/concatenation_all/wrong_and_good2.csv")
