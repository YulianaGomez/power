import pandas as pd
import warnings
import numpy as np
import glob
import os
import sys
from sklearn.preprocessing import StandardScaler



#all data p100 and v100 merged into one dataset
df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/df_master.csv',index_col=0)
df = df.dropna(axis=1,how='any')
df_original = df.copy()
for col in df.columns:
    if not df[col].dtype == "object":

        scaler = StandardScaler().fit(df[col].values.reshape(-1, 1))
        scaled_data_ = scaler.transform(df[col].values.reshape(-1, 1))

        df[col] = scaled_data_

df['unique_id'] =  df['application_name']+df['architecture']+"_"+df['input']
df['semi_unique_id'] =  df['application_name']+"XXXX_"+df['input']
df = df.set_index('unique_id')

df_master_v100 = df[df['architecture'] == "V100"]
df_master_p100 = df[df['architecture'] == "P100"]

df_master_index = df_master_v100.set_index('semi_unique_id').join(
    df_master_p100.set_index('semi_unique_id'),
    lsuffix='_V100',
    rsuffix='_P100'
).dropna(axis=0,how='any').index.values
df_master_index_p100 = [val.replace("XXXX", "P100") for val in df_master_index]
df_master_index_v100 = [val.replace("XXXX", "V100") for val in df_master_index]

df_master_v100 = df[df['architecture'] == "V100"].loc[df_master_index_v100].drop(columns=['semi_unique_id'])
df_master_p100 = df[df['architecture'] == "P100"].loc[df_master_index_p100].drop(columns=['semi_unique_id'])





path = '/Users/yzamora/Desktop/'
p100_index_val = pd.read_csv(path + 'index_val_fullset.txt', names=["vals"]).iloc[:,0]
p100_index_test = pd.read_csv(path + 'index_test_fullset.txt', names=["vals"]).iloc[:,0]
p100_index_train = pd.read_csv(path + 'index_train_fullset.txt', names=["vals"]).iloc[:,0]

v100_index_val = pd.Series([val.replace("P100", "V100") for val in p100_index_val.tolist()])
v100_index_test = pd.Series([val.replace("P100", "V100") for val in p100_index_test.tolist()])
v100_index_train = pd.Series([val.replace("P100", "V100") for val in p100_index_train.tolist()])

# We can simplify the naming a bit to agree with the other naming:
X_val_indices = p100_index_val
y_val_indices = v100_index_val

# How to get a kernel-specific part of the validation set:
X_validation_df = df_master_p100.loc[pd.Series(X_val_indices)]
y_validation_df = df_master_v100.loc[pd.Series(y_val_indices)]

X_validation_sets = {}
X_validation_groups = X_validation_df.groupby("application_name")
for name, group in X_validation_groups:
    X_validation_sets[name] = group.reset_index(drop=True).drop(
        columns = [
            'memory_bound',
            'master_index',
            'architecture',
            'input',
            'application_name',
            'kernelname',
        ]
    ).values

y_validation_sets = {}
y_validation_groups = y_validation_df.groupby("application_name")
for name, group in y_validation_groups:
    #use if you want all 116 metrics
    """y_validation_sets[name] = group.reset_index(drop=True).drop(
        columns = [
            'memory_bound',
            'master_index',
            'architecture',
            'input',
            'application_name',
            'kernelname',
        ]"""
    y_validation_sets[name] = group.reset_index(drop=True)['ipc'].values

# From here, we can get the X validation set for "backprop" by:
#     X_validation_sets["backprop"]
#
# Similarly, we can get the y validation set for "backprop" by:
#     y_validation_sets["backprop"]
groups = ['backprop', 'hybridsort', 'kmeans', 'srad', 'stream', 'gaussian','leukocyte']
for apps in groups:
    print(X_validation_sets[apps].shape, y_validation_sets[apps].shape)

    #print((np.reshape(y_validation_sets[apps],(y_validation_sets[apps].shape[0],1))).shape)
    print(path + '/fullset_specified_apps/'+ apps + '_x_val')
    np.save(path + '/fullset_specified_apps/'+ apps + '_x_val',X_validation_sets[apps])
    np.save(path + '/fullset_specified_apps/'+ apps + '_y_val',np.reshape(y_validation_sets[apps],(y_validation_sets[apps].shape[0],1)))
