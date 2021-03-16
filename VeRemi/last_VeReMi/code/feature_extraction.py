# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, f1_score
from imblearn.under_sampling import RandomUnderSampler

## Get PATH
path = os.getcwd()
database_dir_path = path[:-4].replace(os.sep, "/") + "database/"

## Open the csv
df = pd.read_csv(database_dir_path + "cleandataVeReMi_final.csv")

## Split features and labels
X = df.drop(["attackerType"], axis=1)
y = df["attackerType"]

## Create and fit an undersampling method
undersample = RandomUnderSampler(sampling_strategy='all', random_state=42)
X_under, y_under = undersample.fit_resample(X, y)


## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=y.unique(), y=y.value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type before under sampling\n", fontsize=18, color='#3742fa')
plt.tight_layout()

plt.figure(figsize=(10,6))
sns.barplot(x=y_under.unique(), y=y_under.value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type after under sampling\n", fontsize=18, color='#3742fa')
plt.tight_layout()

## Concat and save in a new csv file the undersampling
df_under = pd.concat([X_under,y_under], axis=1)
df_under.to_csv(database_dir_path + "cleandataVeReMi_final_undersampling.csv")



df1 = pd.read_csv(database_dir_path + "cleandataVeReMi_final_undersampling.csv")

def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 42)
    trainedforest = RandomForestClassifier().fit(X_Train,Y_Train)
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    print(f1_score(Y_Test,predictionforest, average="macro"))

forest_test(X_under,y_under)

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