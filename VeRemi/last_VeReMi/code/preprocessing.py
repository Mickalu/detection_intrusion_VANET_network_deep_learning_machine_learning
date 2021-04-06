# -*- coding: utf-8 -*-

###############################################################################
"""
                            IMPORT OF LIBRARIES
"""
###############################################################################
import os
from os.path import basename
import pandas as pd
from math import sqrt

###############################################################################
"""
                            PREPROCESSING
"""
###############################################################################

## Get PATH
path = os.getcwd()
database_dir_path = path[:-4].replace(os.sep, "/") + "database/"

##################################
""" FUNCTION FOR PREPROCESSING """
##################################
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

## List
dataframe = []
columns_to_split = ["pos", "spd"]
column_to_convert = ["sender", "messageID"]

## Loop to get all the directory and json files that we need to create the csv file
for directory in os.listdir(database_dir_path):
    df_ground = pd.read_json(database_dir_path + directory + "/GroundTruthJSONlog.json", lines=True)
    for file in os.listdir(database_dir_path + directory):
        if os.path.isfile(database_dir_path + directory + "/" + file):
            file_path = database_dir_path + directory + "/" + file
            
            ## Check
            if file != "GroundTruthJSONlog.json":
                df_json = pd.read_json(file_path, lines=True)
    
                ## Delete type 2
                df_without_type2 = delete_type_2(df_json).reset_index()
                
                ## Check if the file contains type 3 (not empty because some contains only type 2)
                if not df_without_type2.empty:
                    
                    ## Convert some columns in int (to get the good format to use)
                    for col in column_to_convert:
                        df_without_type2[col] = df_without_type2[col].astype("int64")    
                    
                    ## Get the receiver ID
                    df_without_type2["receiver"] = int(get_split_filename(file_path))
                    
                    ## Split columns which are using a list to have the data in separate columns
                    for col in columns_to_split:
                        col_x, col_y, col_z = split_column_list(df_without_type2[col])
                        df_without_type2[col + "_x"] = col_x
                        df_without_type2[col + "_y"] = col_y
                        df_without_type2[col + "_z"] = col_z
                        df_without_type2.drop(col, axis=1, inplace=True)
                    
                    ## Calculate the global pos
                    list_pos = []
                    for i in df_without_type2.index:
                        list_pos.append(sqrt(
                            df_without_type2["pos_x"][i] ** 2 + df_without_type2["pos_y"][i] ** 2 + df_without_type2["pos_z"][i] ** 2))
                    df_without_type2["global_pos"] = list_pos
                    
                    ## Calculate the global spd
                    list_spd = []
                    for i in df_without_type2.index:
                        list_spd.append(sqrt(
                            df_without_type2["spd_x"][i] ** 2 + df_without_type2["spd_y"][i] ** 2 + df_without_type2["spd_z"][i] ** 2))
                    df_without_type2["global_spd"] = list_spd
                    
                    ## Check with the "GroundTruthJSONlog.json" to get the attackerType
                    list_attackerType = []
                    for i in df_without_type2.index:
                        row_message = df_ground[df_ground["messageID"] == df_without_type2["messageID"][i]]
                        list_attackerType.append(row_message["attackerType"].values[0])
                    df_without_type2["attackerType"] = list_attackerType
                    
                    ## Add to the list for concatenation
                    dataframe.append(df_without_type2)
                    
## Concatenate all the dataframe create with json files
result = pd.concat(dataframe).drop(["index", "noise", "spd_noise", "pos_noise"], axis=1)

## Save the result in a csv file
result.to_csv(database_dir_path + "cleandataVeReMi.csv", index=False)