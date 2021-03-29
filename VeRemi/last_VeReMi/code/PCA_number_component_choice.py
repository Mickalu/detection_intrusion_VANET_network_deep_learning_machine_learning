import pandas as pd
from function_PCA import *
import seaborn as sns
count = 0

df = pd.read_csv("D:/School/cours_5eme/projet/code/VeRemi/last_VeReMi/database/cleandataVeReMi_with_directory.csv")
liste_number_component = [3,4,5,6,7]
historys = []
matrixs = []
list_col_supp = ['type', 'pos_z', 'spd_z']
list_activation_function = ['tanh', 'relu']

df = delete_column_df(df, list_col_supp)

for activation in list_activation_function:
    for number in liste_number_component:
        history, matrix = value_pca_acc(df, number, activation)
        historys.append(history)
        matrixs.append(matrix)

    accuracy_graph(historys, liste_number_component)
    loss_graph(historys, liste_number_component)

    for index_tab in range(len(matrixs)):
        print(activation, " : \n \n", liste_number_component[index_tab]," : \n ", historys[index_tab].history['accuracy'],"\n", matrixs[index_tab])
