# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 16:54:56 2020

@author: lucas
"""
import pandas as pd
import csv
from os import listdir
from os.path import isfile, join


mypath_before = "D:/School/cours_5eme/projet/database/Kyoto2016/2015/"
liste_folder = ['03','04','05','06','07','08','09','10','11','12']

csv_folder = "/csv_file/"
txt_foler = "/txt_file/"

for number_path in liste_folder: 

    mypath = mypath_before + number_path

    onlyfiles_csv = [f for f in listdir(mypath + csv_folder) if isfile(join(mypath + csv_folder, f))]

    def delete_empty_row(input_file, output_file):    
        with open(input_file) as input, open(output_file, 'w') as output:
            non_blank = (line for line in input if line.strip())
            output.writelines(non_blank)


    for file in onlyfiles_csv:
        delete_empty_row(mypath + csv_folder + file, mypath + csv_folder + "clean_" + file)
        print(file)