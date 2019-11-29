import pandas as pd
import warnings
import numpy as np
import glob
import os
import sys
from sklearn.externals.joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/yzamora/power/nvidia_gpus/all_apps/df_master.csv',index_col=0)
df = df.dropna(axis=1,how='any')
df_original = df.copy()
for col in df.columns:
    if not df[col].dtype == "object":

        scaler = StandardScaler().fit(df[col].values.reshape(-1, 1))
        scaled_data_ = scaler.transform(df[col].values.reshape(-1, 1))

        df[col] = scaled_data_
        dump(scaler, '/Users/yzamora/power/nvidia_gpus/all_apps/large_training_scalers/std_scaler_' + col + '.bin', compress=True)

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

# Check that the indices are aligned
assert(
    df_master_v100.index.values == [
        val.replace("P100", "V100")
        for val in df_master_p100.index
    ]
).all()

#Acquiring 70% of dataset: training, test, validation_data
y = df_master_v100['ipc'].values
X = df_master_p100.drop(columns=['architecture','input','application_name','kernelname', 'memory_bound','master_index']).values
index = df_master_p100.index

X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, index, test_size=.3, random_state=42)
X_val, X_test, y_val, y_test, index_val, index_test = train_test_split(X_test, y_test,index_test, test_size=0.3, random_state=42)

def write_txt(index, name):
    with open ('/Users/yzamora/Desktop/index_' + name + '_fullset.txt','w') as f:
        f.write('\n'.join(list(index)))
    return

np.save('/Users/yzamora/Desktop/X_train_fullset.npy', X_train)
np.save('/Users/yzamora/Desktop/y_train_fullset.npy', y_train)
np.save('/Users/yzamora/Desktop/X_test_fullset.npy', X_test)
np.save('/Users/yzamora/Desktop/y_test_fullset.npy', y_test)
np.save('/Users/yzamora/Desktop/X_val_fullset.npy', X_val)
np.save('/Users/yzamora/Desktop/y_val_fullset.npy', y_val)
write_txt(index_val, "val")
write_txt(index_test, 'test')
write_txt(index_train, 'train')
