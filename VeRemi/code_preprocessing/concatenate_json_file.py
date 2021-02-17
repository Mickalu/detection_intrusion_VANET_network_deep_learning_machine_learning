from os import listdir
from os.path import isfile, join
import pandas as pd
import settings

file_name = "0_3_16_01"
mypath = settings.PATH_folder + "/database/json_file/" + file_name + "/"

list_df_json_concat = []
all_folder_json = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for file in all_folder_json:
    df_json = pd.read_json(mypath + file, lines=True)
    list_df_json_concat.append(df_json)

df_result = pd.concat(list_df_json_concat)

df_result.to_csv(settings.PATH_folder + "/database/csv_file/data_with_noise/" + file_name + "_concatenate.csv")



