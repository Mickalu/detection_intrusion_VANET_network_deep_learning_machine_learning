import pandas as pd

df1 = pd.read_csv("D:/School/cours_5eme/projet/code/database/Kyoto2016/2015/concatenation_all/wrong_and_good2.csv")
df2 = pd.read_csv("D:/School/cours_5eme/projet/code/database/Kyoto2016/2015/concatenation_all/wrong_and_good.csv")

Df_result = df1 + df2


Df_result.to_csv("D:/School/cours_5eme/projet/code/database/Kyoto2016/2015/concatenation_all/wrong_and_good_final.csv")