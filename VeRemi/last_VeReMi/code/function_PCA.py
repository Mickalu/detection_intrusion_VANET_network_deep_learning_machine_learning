import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


def accuracy_graph(historys, liste_number_component):
    count = 0

    for history in historys:
        plt.plot(history.history['accuracy'], label = liste_number_component[count])
        count += 1

    plt.legend(loc = 'upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.show()


def loss_graph(historys,liste_number_component):
    count = 0

    for history in historys:
        plt.plot(history.history['loss'], label = liste_number_component[count])
        count += 1

    plt.legend(loc = 'upper left')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.show()

def value_pca_acc(df, n_component_pca):
    X = df.drop(['attackerType'], axis = 1)
    Y = df['attackerType']
    Y = pd.get_dummies(Y, columns=['attackerType'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


    pca = PCA(n_components= n_component_pca)

    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)



    model=Sequential()

    model.add(Dense(64, input_shape=(n_component_pca,), activation='relu'))
    model.add(Dense(32, activation='relu',))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    history = model.fit(X_train_pca, Y_train, validation_data = (X_test_pca, Y_test), epochs=10)

    Y_predicted = model.predict(X_test_pca)
    matrix = confusion_matrix_pca(Y_predicted, Y_test)
    return history, matrix

def confusion_matrix_pca(Y_predicted, Y_test):
    
    Y_predicted = (Y_predicted > 0.5) 
    matrix = confusion_matrix(Y_test.values.argmax(axis=1),Y_predicted.argmax(axis=1))
    return matrix 

def delete_column_df(df, list_col_supp):
    df = df.drop(list_col_supp, axis=1)
    return df