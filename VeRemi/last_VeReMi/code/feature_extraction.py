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
from sklearn.preprocessing import MinMaxScaler

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


## Open undersampling csv file
df = pd.read_csv(database_dir_path + "cleandataVeReMi_final_undersampling.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

## Feature Selection
df.drop(["type", "pos_z", "spd_z"], axis=1, inplace=True)

## Print correlation matrix
plt.figure(figsize = (16,10))
corrMatrix = df.corr()
ax = sns.heatmap(corrMatrix, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

## Create X and y
X = df.drop("attackerType", axis=1)
y = df["attackerType"]

## Normalize the data
df1 = df
list_columns_to_normalize = ["pos_x", "pos_y", "spd_x", "spd_y", "global_pos", "global_spd"]
for col in list_columns_to_normalize:
    scaler = MinMaxScaler()
    df1[col] = scaler.fit_transform(df1[[col]])
    
X1 = df.drop("attackerType", axis=1)
y1 = df["attackerType"]


## Group by
df_group = df.groupby(by=["sender","receiver","attackerType"], as_index=False).agg({'rcvTime':'mean', 'sendTime':'mean', 'messageID':'first', 'RSSI':'first', 'pos_x':'mean', 'pos_y':'mean', 'pos_z':'mean', 'spd_x':'mean', 'spd_y':'mean',
       'spd_z':'mean', 'global_pos':'mean', 'global_spd':'mean'})


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