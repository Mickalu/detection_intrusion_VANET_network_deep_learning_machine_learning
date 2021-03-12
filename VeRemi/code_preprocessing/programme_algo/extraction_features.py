import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split


from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:/School/cours_5eme/projet/code/VeRemi/database/csv_file/data_without_noise/cleandataVereMi/cleandataVereMi.csv")
X = pd.concat([df['pos_x'], df['pos_y']], axis = 1)

input_layer = Input(shape=(X.shape[1],))

encoded = Dense(X.shape[1], activation='relu')(input_layer)
intermediaire = Dense(7, activation='relu')(encoded)
decoded = Dense(X.shape[1], activation='softmax')(intermediaire)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)

autoencoder.fit(X1, Y1,
                epochs=100,
                batch_size=300,
                shuffle=True,
                verbose = 30,
                validation_data=(X2, Y2))

encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
X_autoencoded = pd.DataFrame(X_ae)

X_autoencoded = pd.DataFrame(X_ae)

XReshape=X_autoencoded.values.reshape((X_autoencoded.values.shape[0], X_autoencoded.values.shape[1], 1))
Y = df.attackerType
X_train, X_test, y_train, y_test = train_test_split(XReshape, Y, test_size = 0.2, random_state = 42)

model=Sequential()

model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.2), input_shape=(2,1), activation='relu'))
model.add(LSTM(32, activation='relu', kernel_regularizer=l2(0.2)))
model.add(Dense(5, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


result = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10)

# accuracy
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

# loss value
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()
