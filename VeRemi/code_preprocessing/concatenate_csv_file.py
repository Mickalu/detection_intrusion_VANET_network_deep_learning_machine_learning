from os import listdir
from os.path import isfile, join
import pandas as pd
import settings

mypath = settings.PATH_folder + "/database/csv_file/data_without_noise/data_clean/"

list_df_csv_concat = []
all_folder_json = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for file in all_folder_json:
    df_csv = pd.read_csv(mypath + file)
    list_df_csv_concat.append(df_csv)

df_result = pd.concat(list_df_csv_concat)

df_result.to_csv(settings.PATH_folder + "/database/csv_file/data_without_noise/bigger_dataframe/concatenate.csv")