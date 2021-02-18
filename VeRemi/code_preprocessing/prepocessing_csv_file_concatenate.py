import pandas as pd
# import seaborn as sns
# from seaborn import *
import numpy as np
import settings
from math import sqrt

folder_type_database = "/database/csv_file/data_with_noise/"
name_file = "0_3_8_01_concatenate.csv"

df_csv = pd.read_csv(settings.PATH_folder + folder_type_database + name_file)


# delete type 2

df_csv_delete_type_2 = df_csv[df_csv['type'] != 2]
df_without_noise = df_csv_delete_type_2.drop(["noise","pos_noise", "spd_noise"], axis = 1)


# Séparer les colonnes str sous forme de liste et ensuite sous forme de colonne

def convert_column_str_to_float(column_trans):
    
    new_column = []

    for element in column_trans:
        element = element[1:-1]
        element = element.split(",")
        element = list(map(float, element))
        new_column.append(element)

    return new_column



## delete columns


list_name_column_str_to_list_float = ["pos","spd"]#"noise"

for col in list_name_column_str_to_list_float:
    
    list_element = convert_column_str_to_float(df_without_noise[col])

    del df_without_noise[col]

    row, column_index = np.shape(list_element)

    for i in range(column_index):
        name_df_new_col = col + "_" + str(i)
        list_element_index = []
        list_element_index = [item[i] for item in list_element]

        df_without_noise[name_df_new_col] = list_element_index


## Ajout colonne type d'attaque

parametre_dataset = name_file.split('_')
colum_attack_type = [int(parametre_dataset[3])] * np.shape(df_without_noise)[0]

df_without_noise['type_attack'] = colum_attack_type


## Ajout colonne pos et speed (général)

list_pos = []
for i in df_without_noise.index:
    list_pos.append(sqrt(df_without_noise['pos_0'][i] ** 2 + df_without_noise['pos_1'][i] ** 2 + df_without_noise['pos_2'][i] ** 2))
df_without_noise['pos'] = list_pos

list_speed = []
for i in df_without_noise.index:
    list_speed.append(sqrt(df_without_noise['spd_0'][i] ** 2 + df_without_noise['spd_1'][i] ** 2 + df_without_noise['spd_2'][i] ** 2))
df_without_noise['speed'] = list_speed

#save result
df_without_noise.to_csv("E:/programmation/projet_5eme/detection_intrusion_VANET_network_deep_learning_machine_learning/VeRemi/database/csv_file/data_without_noise/data_clean/test_" + name_file)