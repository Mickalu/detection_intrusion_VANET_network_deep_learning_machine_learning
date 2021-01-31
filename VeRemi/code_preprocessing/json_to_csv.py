import pandas as pd
PATH = "D:/School/cours_5eme/projet/code/VeRemi/database/0_3_1_01/"
file_name = "JSONlog-0-7-A0.json"

df_json = pd.read_json(PATH + file_name, lines=True)
df_csv = df_json.to_csv()

print(df_csv)