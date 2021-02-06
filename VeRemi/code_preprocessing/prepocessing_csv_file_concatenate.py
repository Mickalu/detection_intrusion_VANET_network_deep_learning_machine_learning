import pandas as pd
# import seaborn as sns
# from seaborn import *
import numpy as np

PATH = "F:/programmation/projet_5eme/detection_intrusion_VANET_network_deep_learning_machine_learning/VeRemi"
df_csv = pd.read_csv(PATH + "/database/csv_file/0_3_1_01_concatenate.csv")

# delete type 2

df_csv_delete_type_2 = df_csv[df_csv['type'] != 2]

# Séparer les colonnes str sous forme de liste et ensuite sous forme de colonne

def convert_column_str_to_float(column_trans):
    
    new_column = []

    for element in column_trans:
        element = element[1:-1]
        element = element.split(",")
        element = list(map(float, element))
        new_column.append(element)

    return new_column





list_name_column_str_to_list_float = ["pos","spd","spd_noise","pos_noise"]#"noise"

for col in list_name_column_str_to_list_float:
    
    list_element = convert_column_str_to_float(df_csv_delete_type_2[col])

    del df_csv_delete_type_2[col]

    row, column_index = np.shape(list_element)

    for i in range(column_index):
        name_df_new_col = col + "_" + str(i)
        list_element_index = []
        list_element_index = [item[i] for item in list_element]

        df_csv_delete_type_2[name_df_new_col] = list_element_index




list_name_colonne_split = ["noise"]

# for col in list_name_colonne_split:
#     list_element = df_csv_delete_type_2[col]

#     del df_csv_delete_type_2[col]

#     row, column_index = np.shape(list_element)

#     for i in range(column_index):
#         name_df_new_col = col + "_" + str(i)
#         list_element_index = []
#         list_element_index = [item[i] for item in list_element]

#         df_csv_delete_type_2[name_df_new_col] = list_element_index

print(df_csv_delete_type_2["noise"])

# # Ajout colonne pour le nom du récepteur

# liste_number = file_name.split('-')

# parametre_dataset = PATH.split('_')
# colum_attack_type = [parametre_dataset[3]] * np.shape(df_csv_delete_type_2)[0]

# df_csv_delete_type_2['type_attack'] = colum_attack_type