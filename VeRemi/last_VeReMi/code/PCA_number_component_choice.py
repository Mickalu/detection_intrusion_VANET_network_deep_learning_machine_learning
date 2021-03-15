import pandas as pd
from function_PCA import *


df = pd.read_csv("D:/School/cours_5eme/projet/code/VeRemi/last_VeReMi/database/cleandataVeReMi_final.csv")
liste_number_component = [2,3,4,5,6]
historys = []
matrixs = []


for number in liste_number_component:
    history, matrix = value_pca_acc(df, number)
    historys.append(history)
    matrixs.append(matrix)

accuracy_graph(historys, liste_number_component)
loss_graph(historys, liste_number_component)

for elem_max in matrixs:
    print(elem_max)