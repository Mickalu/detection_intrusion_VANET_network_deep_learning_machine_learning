# -*- coding: utf-8 -*-

###############################################################################
"""
                            IMPORT OF LIBRARIES
"""
###############################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

###############################################################################
"""
                            WORK ON THE DATASET
"""
###############################################################################

## Get PATH
path = os.getcwd()
database_dir_path = path[:-4].replace(os.sep, "/") + "database/"

## Open the csv
df = pd.read_csv(database_dir_path + "cleandataVeReMi_with_directory.csv")

## Feature Selection
df.drop(["type", "pos_z", "spd_z"], axis=1, inplace=True)

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=df["attackerType"].unique(), 
            y=df["attackerType"].value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type at the beginning\n", fontsize=18, color='#3742fa')
plt.tight_layout()

## Group by
# df_group = df.groupby(by=["sender","receiver", "attackerType", "simulation_directory"], 
#                       as_index=False).agg({'rcvTime':'mean', 'sendTime':'mean', 'RSSI':'mean', 
#                                            'pos_x':'mean', 'pos_y':'mean', 'spd_x':'mean', 'spd_y':'mean',
#                                            'global_pos':'mean', 'global_spd':'mean'})
                                           
## Drop the simulation_directory column
df.drop("simulation_directory", axis=1, inplace=True)

## PCA
X_pos = df[["sender", "receiver", "sendTime", "rcvTime", "pos_x", "pos_y"]]
X_spd = df[["sender", "receiver", "sendTime", "rcvTime", "spd_x", "spd_y"]]

pca_pos = PCA(n_components=1)
X_pca_pos = pca_pos.fit_transform(X_pos)
pca_pos_df = pd.DataFrame(data = X_pca_pos, columns = ['PCA_pos'])

pca_spd = PCA(n_components=1)
X_pca_spd = pca_spd.fit_transform(X_spd)
pca_spd_df = pd.DataFrame(data = X_pca_spd, columns = ['PCA_spd'])

df = pd.concat([df, pca_pos_df, pca_spd_df], axis=1)

## Sampling strategy with SMOTE and RandomUnderSampling
X = df.drop("attackerType", axis=1)
y = df["attackerType"]

# number_strategy_smote = int((y_group.value_counts().max() * 0.5).round())

# strategy = {1:number_strategy_smote, 2:number_strategy_smote, 4:number_strategy_smote, 
#             8:number_strategy_smote, 16:number_strategy_smote}
# oversample = SMOTE(sampling_strategy=strategy, random_state=42)
# undersample = RandomUnderSampler(random_state=42)

# steps = [("o", oversample), ("u", undersample)]
# pipeline = Pipeline(steps=steps)

# X_sample, y_sample = pipeline.fit_resample(X_group,y_group)

oversample = SMOTE(random_state=42)
X_sample, y_sample = oversample.fit_resample(X,y)

## Concatenate the new data
df_sample = pd.concat([X_sample, y_sample], axis=1)

## Plot to see the repartition of labels
plt.figure(figsize=(10,6))
sns.barplot(x=df_sample["attackerType"].unique(), 
            y=df_sample["attackerType"].value_counts().sort_index(), palette="Blues_r")
plt.xlabel('\nAttacker Type', fontsize=15, color='#2980b9')
plt.ylabel("\n", fontsize=15, color='#2980b9')
plt.title("Repartition of attacker type after sampling\n", fontsize=18, color='#3742fa')
plt.tight_layout()

## Rearrange the columns order
df_sample = df_sample[['sender', 'receiver', 'sendTime', 'rcvTime', 'messageID', 'RSSI', 'pos_x', 
         'pos_y', 'spd_x', 'spd_y', 'global_pos','global_spd', 'PCA_pos',
         'PCA_spd', 'attackerType']]

## Print correlation matrix
plt.figure(figsize = (16,10))
corrMatrix = df_sample.corr()
ax = sns.heatmap(corrMatrix, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

## Save in a csv file                          
df_sample.to_csv(database_dir_path + "VeReMi_SMOTE.csv", index=False)


   
## Variance with a global PCA to see how n_components to choose
pca = PCA(n_components=12)
X_pca = pca.fit_transform(X_sample)
explained_variance = pca.explained_variance_
print("Variance for",12,"components :",explained_variance)

plt.plot(explained_variance)
plt.xlabel("# of Features")
plt.ylabel("Variance Explained")
plt.yscale("log")
plt.title("PCA Analysis")
plt.show()

## Create a covariance matrix
covar_matrix = PCA(n_components = 12) 

## Calculate eigenvalues 
covar_matrix.fit(X_sample) 
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios 
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_,decimals=3)*100) 
#cumulative sum of explained variance with 12 features

plt.ylabel('% Variance Explained') 
plt.xlabel('# of Features') 
plt.title('PCA Analysis') 
plt.ylim(99.6,100.2) 
plt.style.context('seaborn-whitegrid')
plt.plot(var)