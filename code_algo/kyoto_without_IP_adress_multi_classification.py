# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:40:50 2021

@author: lucas
"""
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

PATH_FILE = "D:/School/cours_5eme/projet/database/Kyoto2016/2015/01/csv_file/"


df_clean = pd.read_csv(PATH_FILE + "clean_20150101.csv", sep = "\t")


columns_name = ['duration_connection','service','source_bytes','destination_bytes','count','same_srv_rate','serror_rate','srv_serror_rate','dst_host_count','dst_host_srv_count','dst_host_same_src_port_rate','dst_host_serror_rate','dst_host_srv_serror_rate','flag','IDS_detection','malware_detection','ashula_detection','label','source_IP_address','source_port_number','destination_IP_address','destination_port_number','start_time','duration']
df_clean.columns = columns_name

df_clean = df_clean.drop(['destination_IP_address', 'source_IP_address', 'start_time'], axis = 1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

list_encode_columns = ["service", "flag", "malware_detection", "ashula_detection", "duration", "IDS_detection"]

for col in list_encode_columns:
    df_clean[col] = le.fit_transform(df_clean[col])
    


list_onehot_columns = ["service", "flag", "ashula_detection", "duration", "IDS_detection", "malware_detection"]
for column_one_hot in list_onehot_columns:  
    one_hot = pd.get_dummies(df_clean[column_one_hot], prefix = column_one_hot)
    # Drop column B as it is now encoded
    df_clean = df_clean.drop(column_one_hot,axis = 1)
    # Join the encoded df
    df_clean = df_clean.join(one_hot)
    
df_clean = df_clean.astype(int)

list_column_malware = []
for col in df_clean.columns:
    if "malware_detection" in col:
        list_column_malware.append(col)
        
X = df_clean.drop(list_column_malware, axis = 1)
y = df_clean[list_column_malware]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Question 1
# instantiate the model, add hidden and output layers
model=Sequential()
model.add(Dense(64, input_shape=(117,), activation='tanh'))
model.add(Dense(45, activation='sigmoid'))

# Question 2
# compile and summarize the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


# train the model 
h = model.fit(X_train, y_train,validation_data = (X_test, y_test), epochs=30)


def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model multi classification loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

# Question 7
# plot train and test losses
plot_loss(h.history['loss'], h.history['val_loss'])