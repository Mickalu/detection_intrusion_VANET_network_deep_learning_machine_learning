from os import listdir
from os.path import isfile, join
import pandas as pd

mypath = "F:/programmation/projet_5eme/detection_intrusion_VANET_network_deep_learning_machine_learning/VeRemi/database/json_file/0_3_1_01/"
list_df_json_concat = []
all_folder_json = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for file in all_folder_json:
    df_json = pd.read_json(mypath + file, lines=True)
    list_df_json_concat.append(df_json)

df_result = pd.concat(list_df_json_concat)

df_result.to_csv("F:/programmation/projet_5eme/detection_intrusion_VANET_network_deep_learning_machine_learning/VeRemi/database/CSV_file/0_3_1_01_concatenate.csv")