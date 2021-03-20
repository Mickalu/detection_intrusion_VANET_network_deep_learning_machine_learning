import pandas as pd
from function_PCA import *
import seaborn as sns
count = 0

df = pd.read_csv("D:/School/cours_5eme/projet/code/VeRemi/last_VeReMi/database/cleandataVeReMi_final_undersampling.csv", index = False)
liste_number_component = [2,3,4,5,6]
historys = []
matrixs = []
list_col_supp = ['type', 'pos_z', 'spd_z', 'Unnamed: 0']
list_activation_function = ['tanh', 'relu']

df = delete_column_df(df, list_col_supp)

for number in liste_number_component:
    history, matrix = value_pca_acc(df, number, 'relu')
    historys.append(history)
    matrixs.append(matrix)

accuracy_graph(historys, liste_number_component)
loss_graph(historys, liste_number_component)

for matrix in matrixs:
    print(matrix)
    plt.figure(figsize = (10,7))
    sns.heatmap(matrix, annot=True)
    count += 1