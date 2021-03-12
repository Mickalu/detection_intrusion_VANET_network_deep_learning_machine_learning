# -*- coding: utf-8 -*-

import os
from os.path import basename
import pandas as pd
from math import sqrt

path = os.getcwd()

database_dir_path = path[:-4].replace(os.sep, "/") + "database/"

def delete_type_2(dataFrame):
    return dataFrame[(dataFrame["type"] != 2)]


def get_split_filename(file_path):
    fileName, fileExtension = os.path.splitext(basename(file_path))
    fileName_split = fileName.split("-")
    return fileName_split[-2]


def split_column_list(column_to_split):
    new_column_x = []
    new_column_y = []
    new_column_z = []

    for element in column_to_split:
        new_column_x.append(element[0])
        new_column_y.append(element[1])
        new_column_z.append(element[2])

    return new_column_x, new_column_y, new_column_z


dataframe = []
columns_to_split = ["pos", "spd"]
column_to_convert = ["sender", "messageID"]

for directory in os.listdir(database_dir_path):
    df_ground = pd.read_json(database_dir_path + directory + "/GroundTruthJSONlog.json", lines=True)
    for file in os.listdir(database_dir_path + directory):
        if os.path.isfile(database_dir_path + directory + "/" + file):
            file_path = database_dir_path + directory + "/" + file

            if file != "GroundTruthJSONlog.json":
                df_json = pd.read_json(file_path, lines=True)
    
                df_without_type2 = delete_type_2(df_json).reset_index()
                
                if not df_without_type2.empty:
                
                    for col in column_to_convert:
                        df_without_type2[col] = df_without_type2[col].astype("int64")    
                
                    df_without_type2["receiver"] = int(get_split_filename(file_path))
        
                    for col in columns_to_split:
                        col_x, col_y, col_z = split_column_list(df_without_type2[col])
                        df_without_type2[col + "_x"] = col_x
                        df_without_type2[col + "_y"] = col_y
                        df_without_type2[col + "_z"] = col_z
                        df_without_type2.drop(col, axis=1, inplace=True)
        
                    list_pos = []
                    for i in df_without_type2.index:
                        list_pos.append(sqrt(
                            df_without_type2["pos_x"][i] ** 2 + df_without_type2["pos_y"][i] ** 2 + df_without_type2["pos_z"][i] ** 2))
                    df_without_type2["global_pos"] = list_pos
        
                    list_spd = []
                    for i in df_without_type2.index:
                        list_spd.append(sqrt(
                            df_without_type2["spd_x"][i] ** 2 + df_without_type2["spd_y"][i] ** 2 + df_without_type2["spd_z"][i] ** 2))
                    df_without_type2["global_spd"] = list_spd
                    
                    
                    list_attackerType = []
                    for i in df_without_type2.index:
                        row_message = df_ground[df_ground["messageID"] == df_without_type2["messageID"][i]]
                        list_attackerType.append(row_message["attackerType"].values[0])
                    df_without_type2["attackerType"] = list_attackerType
                    
                    dataframe.append(df_without_type2)

result = pd.concat(dataframe).drop(["index", "noise", "spd_noise", "pos_noise"], axis=1)

result.to_csv(database_dir_path + "cleandataVeReMi.csv", index=False)