# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:01:28 2021

@author: lucas
"""
import matplotlib.pyplot as plt

def plot_loss(loss,val_loss, title):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title(title)
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()
  
def plot_accuracy(accuracy, val_accuracy, title):
  plt.figure()
  plt.plot(accuracy)
  plt.plot(val_accuracy)
  plt.title(title)
  plt.ylabel('accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()
  
def graph_tab(tab, title, y_title):
  plt.figure()
  
  for elem in tab:
      plt.scatter(elem)
      
  plt.title(title)
  plt.ylabel(y_title)
  plt.xlabel('Epoch')
  plt.show()