# -*- coding: utf-8 -*-

###############################################################################
"""
                            IMPORT OF LIBRARIES
"""
###############################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


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
df.drop(["type", "pos_z", "spd_z"], axis=1, inplace=True)

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=df["attackerType"].unique(), 
            y=df["attackerType"].value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type at the beginning\n", fontsize=18, color='#3742fa')
plt.tight_layout()

## Group by
# df_group = df.groupby(by=["sender","receiver", "attackerType", "simulation_directory"], 
#                       as_index=False).agg({'rcvTime':'mean', 'sendTime':'mean', 'RSSI':'mean', 
#                                            'pos_x':'mean', 'pos_y':'mean', 'spd_x':'mean', 'spd_y':'mean',
#                                            'global_pos':'mean', 'global_spd':'mean'})
                                           
## Drop the simulation_directory column
df.drop("simulation_directory", axis=1, inplace=True)

## PCA
X_pos = df[["pos_x", "pos_y"]]
X_spd = df[["spd_x", "spd_y"]]

pca_pos = PCA(n_components=1)
X_pca_pos = pca_pos.fit_transform(X_pos)
pca_pos_df = pd.DataFrame(data = X_pca_pos, columns = ['PCA_pos'])

pca_spd = PCA(n_components=1)
X_pca_spd = pca_spd.fit_transform(X_spd)
pca_spd_df = pd.DataFrame(data = X_pca_spd, columns = ['PCA_spd'])

df = pd.concat([df, pca_pos_df, pca_spd_df], axis=1)

## Sampling strategy with SMOTE and RandomUnderSampling
X = df.drop("attackerType", axis=1)
y = df["attackerType"]

# number_strategy_smote = int((y_group.value_counts().max() * 0.5).round())

# strategy = {1:number_strategy_smote, 2:number_strategy_smote, 4:number_strategy_smote, 
#             8:number_strategy_smote, 16:number_strategy_smote}
# oversample = SMOTE(sampling_strategy=strategy, random_state=42)
# undersample = RandomUnderSampler(random_state=42)

# steps = [("o", oversample), ("u", undersample)]
# pipeline = Pipeline(steps=steps)

# X_sample, y_sample = pipeline.fit_resample(X_group,y_group)

oversample = SMOTE(random_state=42)
X_sample, y_sample = oversample.fit_resample(X,y)


## Concatenate the new data
df_sample = pd.concat([X_sample, y_sample], axis=1)

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=df_sample["attackerType"].unique(), 
            y=df_sample["attackerType"].value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type after sampling\n", fontsize=18, color='#3742fa')
plt.tight_layout()

## Normalize the data
list_columns_to_normalize = ["pos_x", "pos_y", "spd_x", "spd_y", "global_pos", "global_spd"]
for col in list_columns_to_normalize:
    scaler = MinMaxScaler()
    X_sample[col] = scaler.fit_transform(X_sample[[col]])

## Rearrange the columns order
df_sample = df_sample[['sender', 'receiver', 'sendTime', 'rcvTime', 'messageID', 'RSSI', 'pos_x', 
         'pos_y', 'spd_x', 'spd_y', 'global_pos','global_spd', 'PCA_pos',
         'PCA_spd', 'attackerType']]

## Print correlation matrix
plt.figure(figsize = (16,10))
corrMatrix = df_sample.corr()
ax = sns.heatmap(corrMatrix, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

## Save in a csv file                          
df_sample.to_csv(database_dir_path + "VeReMi_SMOTE.csv", index=False)

## Open csv
df_sample = pd.read_csv("C:/Users/Thomas/Desktop/VeReMi_SMOTE.csv")
df_sample.drop(["global_pos", "PCA_spd"], axis=1, inplace=True)

## Split features and labels
X_sample = df_sample.drop("attackerType", axis=1)
# [["sender", "receiver", "sendTime", "rcvTime", "messageID", "RSSI", "PCA_pos", "PCA_spd"]]
y_sample = df_sample["attackerType"]

## Create One Hot Encoding
le = LabelEncoder()
y_le = le.fit_transform(y_sample)
y_ohe = to_categorical(y_le)
   
## Variance with a global PCA to see how n_components to choose
pca = PCA(n_components=12)
X_pca = pca.fit_transform(X_sample)
explained_variance = pca.explained_variance_
print("Variance for",12,"components :",explained_variance)

plt.plot(explained_variance)
plt.xlabel("# of Features")
plt.ylabel("Variance Explained")
plt.yscale("log")
plt.title("PCA Analysis")
plt.show()

#Créer une matrice de covariance 
covar_matrix = PCA(n_components = 12) #vous avez 14 features 

#Calculer les valeurs propres 
covar_matrix.fit(X_sample) 
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios 
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_,decimals=3)*100) 
var #somme cumulative de variance expliquée avec [n] features 

#afficher sur une courbe 
plt.ylabel('% Variance Explained') 
plt.xlabel('# of Features') 
plt.title('PCA Analysis') 
plt.ylim(99.6,100.2) 
plt.style.context('seaborn-whitegrid')
plt.plot(var)


###############################################################################
"""
                            TEST FOR THE MODEL
"""
###############################################################################

def deep_learning_model_3_layer(X,y,test_size,first_layer,second_layer,third_layer,hidden_activation,output_activation,epochs):

    ## Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = test_size, 
                                                    random_state = 42)

    ## Deep Learning model
    model = Sequential()

    model.add(Dense(first_layer, input_shape=(X.shape[1],), activation=hidden_activation))
    model.add(Dense(second_layer, activation=hidden_activation))
    model.add(Dense(third_layer, activation=hidden_activation))
    model.add(Dense(6, activation=output_activation))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=8192)

    ## Prediction and metrics
    y_pred = model.predict_classes(X_test)
    y_pred = le.inverse_transform(y_pred)

    y_test = np.argmax(y_test, axis=1)
    y_test = le.inverse_transform(y_test)

    confusion_matrix_model = confusion_matrix(y_test, y_pred)
    classification_report_model = classification_report(y_test, y_pred)
    accuracy_score_model = accuracy_score(y_test, y_pred)
    f1_score_model = f1_score(y_test, y_pred, average="macro")
    print(confusion_matrix_model)
    print(classification_report_model)
    print("Accuracy score :", accuracy_score_model)
    print("F1 score (macro) :", f1_score_model)
    print("Test size :", test_size)
    print("First layer :", first_layer)
    print("Second layer :", second_layer)
    print("Third layer :", third_layer)
    print("Activation function hidden layer :", hidden_activation)
    print("Activation function output layer :", output_activation)
    print("Epochs :", epochs)
    return [confusion_matrix_model, classification_report_model, accuracy_score_model, 
            f1_score_model, test_size, first_layer, second_layer, third_layer, hidden_activation, 
            output_activation, epochs]
 
def deep_learning_model_4_layer(X,y,test_size,first_layer,second_layer,third_layer,fourth_layer,hidden_activation,output_activation,epochs):

    ## Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = test_size, 
                                                    random_state = 42)

    ## Deep Learning model
    model = Sequential()

    model.add(Dense(first_layer, input_shape=(X.shape[1],), activation=hidden_activation))
    model.add(Dense(second_layer, activation=hidden_activation))
    model.add(Dense(third_layer, activation=hidden_activation))
    model.add(Dense(fourth_layer, activation=hidden_activation))
    model.add(Dense(6, activation=output_activation))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=8192)

    ## Prediction and metrics
    y_pred = model.predict_classes(X_test)
    y_pred = le.inverse_transform(y_pred)

    y_test = np.argmax(y_test, axis=1)
    y_test = le.inverse_transform(y_test)

    confusion_matrix_model = confusion_matrix(y_test, y_pred)
    classification_report_model = classification_report(y_test, y_pred)
    accuracy_score_model = accuracy_score(y_test, y_pred)
    f1_score_model = f1_score(y_test, y_pred, average="macro")
    print(confusion_matrix_model)
    print(classification_report_model)
    print("Accuracy score :", accuracy_score_model)
    print("F1 score (macro) :", f1_score_model)
    print("Test size :", test_size)
    print("First layer :", first_layer)
    print("Second layer :", second_layer)
    print("Third layer :", third_layer)
    print("Fourth layer :", fourth_layer)
    print("Activation function hidden layer :", hidden_activation)
    print("Activation function output layer :", output_activation)
    print("Epochs :", epochs)
    return [confusion_matrix_model, classification_report_model, accuracy_score_model, 
            f1_score_model, test_size, first_layer, second_layer, third_layer, fourth_layer, 
            hidden_activation, output_activation, epochs]
    
first, second, third = [12,13,14]

## Loop to test ANN model
test_size_list = [0.2, 0.25, 0.3]
hidden_activation_list = ["relu", "tanh"]
output_activation_list = ["softmax", "sigmoid"]
hidden_layer_list_3 = [[64,32,16], [128,64,32], [256,128,64], [512,256,128]]
hidden_layer_list_4 = [[]]

results = []

## For 3 layers
for test_size in test_size_list:
    for n in range(2,10,1):
        pca = PCA(n_components=n, random_state=42)
        X_pca = pca.fit_transform(X_sample)
        for hidden_layer in hidden_layer_list_3:
            first_layer, second_layer, third_layer = hidden_layer
            for hidden_activation in hidden_activation_list:
                for output_activation in output_activation_list:
                    result = deep_learning_model_3_layer(X_pca,y_ohe,test_size,first_layer,
                                                second_layer,third_layer,hidden_activation,
                                                output_activation,100)
                    result.append(n)
                    results.append(result)
                    print("PCA n_components :", n)
              

## For 4 layers
for test_size in test_size_list:
    for n in range(2,10,1):
        pca = PCA(n_components=n, random_state=42)
        X_pca = pca.fit_transform(X_sample)
        for hidden_layer in hidden_layer_list_4:
            first_layer, second_layer, third_layer, fourth_layer = hidden_layer
            for hidden_activation in hidden_activation_list:
                for output_activation in output_activation_list:
                    result = deep_learning_model_4_layer(X_pca,y_ohe,test_size,first_layer,
                                                second_layer,third_layer,fourth_layer,hidden_activation,
                                                output_activation,100)
                    result.append(n)
                    results.append(result)
                    print("PCA n_components :", n)