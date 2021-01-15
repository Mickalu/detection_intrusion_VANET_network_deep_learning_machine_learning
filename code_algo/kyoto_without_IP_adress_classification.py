# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:50:20 2021

@author: lucas
"""
from plot_function import *
from keras.models import Sequential
from keras.layers import Dense


from sklearn.model_selection import train_test_split
import pandas as pd


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
    
df_clean.loc[df_clean['malware_detection'] != 0] = 1


list_onehot_columns = ["service", "flag", "ashula_detection", "duration", "IDS_detection"]
for column_one_hot in list_onehot_columns:  
    one_hot = pd.get_dummies(df_clean[column_one_hot], prefix = column_one_hot)
    # Drop column B as it is now encoded
    df_clean = df_clean.drop(column_one_hot,axis = 1)
    # Join the encoded df
    df_clean = df_clean.join(one_hot)
    
df_clean = df_clean.astype(int)


################## algo ####################################


X = df_clean.drop(['malware_detection'], axis = 1)
y = df_clean[['malware_detection']]

list_test_size = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
list_loss = []
list_acc = []


for test_sizes in list_test_size:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sizes, random_state = 42)
    
    
    
    model=Sequential()
    model.add(Dense(64, input_shape=(107,), activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    
     
    h = model.fit(X_train, y_train,validation_data = (X_test, y_test), epochs=15)
    
    
    
    # plot_loss(h.history['loss'], h.history['val_loss'], "loss graph classification model" + str(test_sizes))
    # plot_accuracy(h.history['accuracy'], h.history['val_accuracy'], "accuracy graph classification model" + str(test_sizes))
    
    list_loss.append(h.history['val_loss'])
    list_acc.append(h.history['val_accuracy'])
    
