import pandas as pd
from function_PCA import *
count = 0

df = pd.read_csv("D:/School/cours_5eme/projet/code/VeRemi/last_VeReMi/database/cleandataVeReMi_final_undersampling.csv")
liste_number_component = [2,3,4,5,6]
historys = []
matrixs = []
list_col_supp = ['type', 'pos_z', 'spd_z']

df = delete_column_df(df, list_col_supp)

for number in liste_number_component:
    history, matrix = value_pca_acc(df, number)
    historys.append(history)
    matrixs.append(matrix)

accuracy_graph(historys, liste_number_component)
loss_graph(historys, liste_number_component)

for elem_max in matrixs:
    print(liste_number_component[count]," : " , elem_max)
    count += 1