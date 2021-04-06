import pandas as pd
import csv
from os import listdir
from os.path import isfile, join

import csv
from os import listdir
from os.path import isfile, join

mypath_before = "D:/School/cours_5eme/projet/database/Kyoto2016/2015/"
liste_folder = ['01','02','03','04','05','06','07','08','09','10','11','12']

csv_folder = "/csv_file/"
txt_foler = "/txt_file/"

for number_path in liste_folder: 

    mypath = mypath_before + number_path
    onlyfiles_txt = [f for f in listdir(mypath + txt_foler) if isfile(join(mypath + txt_foler, f))]
    onlyfiles_csv = [f for f in listdir(mypath + csv_folder) if isfile(join(mypath + csv_folder, f))]

    def delete_empty_row(input_file, output_file):    
        with open(input_file) as input, open(output_file, 'w') as output:
            non_blank = (line for line in input if line.strip())
            output.writelines(non_blank)
                
                
    def convert_txt_to_csv(file, mypath):
        csv_file_name = file[:-4] + ".csv"
        
        with open(mypath + txt_foler + file, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            
            
            with open(mypath + csv_folder + csv_file_name, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(lines)
                out_file.close()
                in_file.close()
                
        # with open(mypath + csv_folder + csv_file_name) as input, open(mypath + csv_folder + "clean_" + csv_file_name , 'w') as output:
        #     non_blank = (line for line in input if line.strip())
        #     output.writelines(non_blank)
        #     # output.close()
        

    for file in onlyfiles_txt:
        convert_txt_to_csv(file, mypath)
        
    for file in onlyfiles_csv:
        print(file)
        delete_empty_row(mypath + csv_folder + file, mypath + csv_folder + "clean_" + file)