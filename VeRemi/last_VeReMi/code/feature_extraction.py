# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, f1_score

path = os.getcwd()

database_dir_path = path[:-4].replace(os.sep, "/") + "database/"

df = pd.read_csv(database_dir_path + "cleandataVeReMi.csv")

j=0
for i in range(df.shape[0]):
    if df["attackerType"][i] == 0:
        if j<1000:
            j+=1
        else:
            df.drop(i, axis=0, inplace=True)

X = df.drop(["attackerType"], axis=1)
y = df["attackerType"]

def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 42)
    trainedforest = RandomForestClassifier().fit(X_Train,Y_Train)
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    print(f1_score(Y_Test,predictionforest, average="macro"))

forest_test(X,y)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, df['attackerType']], axis = 1)
# PCA_df.head()

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

classes = [0, 1, 2, 4, 8, 16]
for clas in classes:
    plt.scatter(PCA_df.loc[PCA_df['attackerType'] == clas, 'PC1'], 
                PCA_df.loc[PCA_df['attackerType'] == clas, 'PC2'])
    
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 15)
plt.legend(['Normal', 'Attack_1', 'Attack_2', 'Attack_4','Attack_8', 'Attack_16'])
plt.grid()

forest_test(X_pca, y)

###############################################################################

def len_array_atacker_type_value(value):
    result = len(df[df["attackerType"] == value])
    return result

attackTypeArray = [0, 1, 2, 4, 8, 16]

for attack in attackTypeArray:
    print(attack, " : ", len_array_atacker_type_value(attack), "\n")