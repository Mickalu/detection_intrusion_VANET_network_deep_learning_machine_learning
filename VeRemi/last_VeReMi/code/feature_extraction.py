# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import classification_report,confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

## Get PATH
path = os.getcwd()
database_dir_path = path[:-4].replace(os.sep, "/") + "database/"

## Open the csv
df = pd.read_csv(database_dir_path + "cleandataVeReMi_with_directory.csv")

## Feature Selection
df.drop(["type", "messageID", "pos_z", "spd_z"], axis=1, inplace=True)

## Group by
df_group = df.groupby(by=["sender","receiver","attackerType", "simulation_directory"], 
                      as_index=False).agg({'rcvTime':'mean', 'sendTime':'mean', 'RSSI':'first', 
                                           'pos_x':'mean', 'pos_y':'mean', 'spd_x':'mean', 'spd_y':'mean',
                                           'global_pos':'mean', 'global_spd':'mean'})
                                           
## Drop the simulation_directory column
df_group.drop("simulation_directory", axis=1, inplace=True)

## Rearrange the columns order
df_group = df_group[['sender', 'receiver', 'rcvTime',
       'sendTime', 'RSSI', 'pos_x', 'pos_y', 'spd_x', 'spd_y', 'global_pos',
       'global_spd', 'attackerType']]

## Sampling strategy with SMOTE and RandomUnderSampling
X_group = df_group.drop("attackerType", axis=1)
y_group = df_group["attackerType"]

number_strategy_smote = int((y_group.value_counts().max() * 0.5).round())

strategy = {1:number_strategy_smote, 2:number_strategy_smote, 4:number_strategy_smote, 8:number_strategy_smote, 16:number_strategy_smote}
oversample = SMOTE(sampling_strategy=strategy, random_state=42)
undersample = RandomUnderSampler(random_state=42)

steps = [("o", oversample), ("u", undersample)]
pipeline = Pipeline(steps=steps)

X_sample, y_sample = pipeline.fit_resample(X_group,y_group)

## Concatenate the new data
df_sample = pd.concat([X_sample, y_sample], axis=1)

## Create One Hot Encoding
df_sample = pd.concat([df_sample,pd.get_dummies(df_sample['attackerType'], prefix='attackerType')],axis=1).drop(['attackerType'],axis=1)

## Normalize the data
list_columns_to_normalize = ["pos_x", "pos_y", "spd_x", "spd_y", "global_pos", "global_spd"]
for col in list_columns_to_normalize:
    scaler = MinMaxScaler()
    df_sample[col] = scaler.fit_transform(df_sample[[col]])
                                            
## Split features and labels
list_attackerType = ["attackerType_0", "attackerType_1", "attackerType_2", "attackerType_4", "attackerType_8", "attackerType_16"]
X_sample = df_sample.drop(list_attackerType, axis=1)
y_sample = df_sample[list_attackerType]

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=y.unique(), y=y.value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type before under sampling\n", fontsize=18, color='#3742fa')
plt.tight_layout()

## Print correlation matrix
plt.figure(figsize = (16,10))
corrMatrix = df.corr()
ax = sns.heatmap(corrMatrix, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()


def lstm_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    lstm_model = Sequential()

    lstm_model.add(Dense(64, input_shape=(X.shape[1],)))
    lstm_model.add(Dense(32))
    # lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = 11))
    # lstm_model.add(Dropout(0.2))

    # lstm_model.add(LSTM(units = 50))
    # lstm_model.add(Dropout(0.2))

    lstm_model.add(Dense(units = 6, activation='softmax'))

    lstm_model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    lstm_model.fit(X_train, y_train, epochs = 10)
    
    y_predicted = lstm_model.predict(X_test)
    y_predicted = y_predicted.round()
    print(confusion_matrix(y_test.argmax(axis=1), y_predicted.argmax(axis=1)))
    print(classification_report(y_test.argmax(axis=1), y_predicted.argmax(axis=1)))
    print(f1_score(y_test.argmax(axis=1), y_predicted.argmax(axis=1), average="macro"))


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