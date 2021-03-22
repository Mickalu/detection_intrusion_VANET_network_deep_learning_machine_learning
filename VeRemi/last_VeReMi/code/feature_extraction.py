# -*- coding: utf-8 -*-

## Import
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report,multilabel_confusion_matrix, f1_score


###############################################################################
"""
                            WORK ON THE DATASET
"""
###############################################################################

## Get PATH
path = os.getcwd()
database_dir_path = path[:-4].replace(os.sep, "/") + "database/"

## Open the csv
df = pd.read_csv(database_dir_path + "cleandataVeReMi_with_directory.csv")

## Feature Selection
df.drop(["type", "messageID", "pos_z", "spd_z"], axis=1, inplace=True)

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=df["attackerType"].unique(), y=df["attackerType"].value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type at the beginning\n", fontsize=18, color='#3742fa')
plt.tight_layout()

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

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=df_group["attackerType"].unique(), y=df_group["attackerType"].value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type with group_by\n", fontsize=18, color='#3742fa')
plt.tight_layout()

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

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=df_sample["attackerType"].unique(), y=df_sample["attackerType"].value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type after sampling\n", fontsize=18, color='#3742fa')
plt.tight_layout()

## Normalize the data
list_columns_to_normalize = ["pos_x", "pos_y", "spd_x", "spd_y", "global_pos", "global_spd"]
for col in list_columns_to_normalize:
    scaler = MinMaxScaler()
    df_sample[col] = scaler.fit_transform(df_sample[[col]])

## Create One Hot Encoding
df_sample = pd.concat([df_sample,pd.get_dummies(df_sample['attackerType'], prefix='attackerType')],axis=1).drop(['attackerType'],axis=1)
    
## Print correlation matrix
plt.figure(figsize = (16,10))
corrMatrix = df_sample.corr()
ax = sns.heatmap(corrMatrix, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
                                            
## Split features and labels
list_attackerType = ["attackerType_0", "attackerType_1", "attackerType_2", "attackerType_4", "attackerType_8", "attackerType_16"]
X_sample = df_sample.drop(list_attackerType, axis=1)
y_sample = df_sample[list_attackerType]


###############################################################################
"""
                            TEST FOR THE MODEL
"""
###############################################################################

## Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size = 0.25, shuffle=False, random_state = 42)


## Deep Learning model
lstm_model = Sequential()

lstm_model.add(Dense(20, input_shape=(X_sample.shape[1],)))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(20))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(10))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(6, activation='softmax'))

lstm_model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

lstm_model.fit(X_train, y_train, epochs = 10)

## Prediction and metrics
y_predicted = lstm_model.predict(X_test)
y_predicted = y_predicted.round()
print(multilabel_confusion_matrix(y_test, y_predicted))
print(classification_report(y_test, y_predicted))
print(f1_score(y_test, y_predicted, average="macro"))