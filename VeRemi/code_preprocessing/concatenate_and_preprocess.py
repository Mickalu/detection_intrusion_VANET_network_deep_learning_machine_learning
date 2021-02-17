from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

name_file_data = "0_3_8_01"
PATH = "E:/programmation/projet_5eme/detection_intrusion_VANET_network_deep_learning_machine_learning/VeRemi/database/"
folder_data = "data_without_noise/"

mypath = PATH + "json_file/"+ name_file_data +"/"

list_df_json_concat = []
all_folder_json = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for file in all_folder_json:
    df_json = pd.read_json(mypath + file, lines=True)
    list_df_json_concat.append(df_json)

df_result = pd.concat(list_df_json_concat)



# delete type 2

df_csv_delete_type_2 = df_result[df_result['type'] != 2]
del df_csv_delete_type_2["noise"]
del df_csv_delete_type_2["pos_noise"]

# Séparer les colonnes str sous forme de liste et ensuite sous forme de colonne

def convert_column_str_to_float(column_trans):
    
    new_column = []

    for element in column_trans:
        element = element[1:-1]
        element = element.split(",")
        element = list(map(float, element))
        new_column.append(element)

    return new_column





list_name_column_str_to_list_float = ["pos","spd","spd_noise"]#"noise"

for col in list_name_column_str_to_list_float:
    
    list_element = convert_column_str_to_float(df_csv_delete_type_2[col])

    del df_csv_delete_type_2[col]

    row, column_index = np.shape(list_element)

    for i in range(column_index):
        name_df_new_col = col + "_" + str(i)
        list_element_index = []
        list_element_index = [item[i] for item in list_element]

        df_csv_delete_type_2[name_df_new_col] = list_element_index



## Ajout colonne type d'attaque


parametre_dataset = name_file.split('_')
colum_attack_type = [int(parametre_dataset[3])] * np.shape(df_csv_delete_type_2)[0]

df_csv_delete_type_2['type_attack'] = colum_attack_type


#save result
df_csv_delete_type_2.to_csv(PATH+ folder_data+ "csv_file/clean_data/" + name_file+".csv")