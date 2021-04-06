# -*- coding: utf-8 -*-

###############################################################################
"""
                            IMPORT OF LIBRARIES
"""
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


###############################################################################
"""
                            TEST FOR THE MODEL
"""
###############################################################################

## Open csv
df_sample = pd.read_csv("C:/Users/Thomas/Desktop/VeReMi_SMOTE.csv")
df_sample.drop(["global_pos", "PCA_spd"], axis=1, inplace=True)

## Split features and labels
X_sample = df_sample.drop("attackerType", axis=1)
y_sample = df_sample["attackerType"]

## Create One Hot Encoding
le = LabelEncoder()
y_le = le.fit_transform(y_sample)
y_ohe = to_categorical(y_le)

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

def deep_learning_model_lstm(X,y,epochs):
    n_comps = X.shape[1]
    
    ## Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)
    
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    ## Deep Learning model
    model_lstm = Sequential()

    model_lstm.add(LSTM(units=n_comps, return_sequences=False,
               input_shape=(X_train.shape[1], 1),
               dropout=0.2, recurrent_dropout=0.2))
    model_lstm.add(Dense(64, activation="relu"))
    model_lstm.add(Dense(32, activation="relu"))
    model_lstm.add(Dense(6, activation="softmax"))

    model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_lstm.fit(X_train, y_train, epochs=epochs, batch_size=8192)

    ## Prediction and metrics
    y_pred = model_lstm.predict_classes(X_test)
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
    print("Epochs :", epochs)



## Loop to test ANN model
test_size_list = [0.2, 0.25, 0.3]
hidden_activation_list = ["relu", "tanh"]
output_activation_list = ["softmax", "sigmoid"]
hidden_layer_list_2 = [[256,128]]
hidden_layer_list_3 = [[64,32,16], [128,64,32], [256,128,64], [512,256,128]]
hidden_layer_list_4 = [[256,128,64,32]]

results = []

for n in range(2,10,1):
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    result = deep_learning_model_3_layer(X_pca,y_ohe,0.2,128,
                                                64,32,"relu",
                                                "softmax",100)
    result.append(n)
    results.append(result)
    print("PCA n_components :", n)
    
x_components = [r[-1] for r in results]
y_components = [r[2] for r in results]

plt.plot(x_components,y_components)
plt.xlabel("Number of components")
plt.ylabel("Accuracy")
plt.show()

pca = PCA(n_components=7, random_state=42)
X_pca = pca.fit_transform(X_sample)

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