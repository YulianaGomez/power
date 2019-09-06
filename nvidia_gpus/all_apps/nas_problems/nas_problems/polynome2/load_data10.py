import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

def load_data():
    path = '/home/yzamora/power_update/nvidia_gpus/all_apps/'
    limited_data = False
    npy_files = False
    specified_data = False # 20 percent with AL
    per_10 = True #10 percent selected randomly
    if per_10:
        X_train = np.load(path + 'X_train_RANDOM_10per.npy')
        X_test = np.load(path + 'X_val_RANDOM_10per.npy')
        y_train = np.load(path + 'y_train_RANDOM_10per.npy')
        y_test = np.load(path + 'y_val_RANDOM_10per.npy')
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return (X_train, y_train), (X_test, y_test)
    if specified_data:
        X_train = np.load(path + 'X_train_RANDOM_20per.npy')
        X_test = np.load(path + 'X_val_RANDOM_20per.npy')
        y_train = np.load(path + 'y_train_RANDOM_20per.npy')
        y_test = np.load(path + 'y_val_RANDOM_20per.npy')
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return (X_train, y_train), (X_test, y_test)
    if npy_files:
        X_train = np.load(path + 'X_train_DH.npy')
        X_test = np.load(path + 'X_test_DH.npy')
        y_train = np.load(path + 'y_train_DH.npy')
        y_test = np.load(path + 'y_test_DH.npy')
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return (X_train, y_train), (X_test, y_test)
    elif limited_data:
        path = '/home/yzamora/power_update/nvidia_gpus/all_apps/'
        p100_20per = pd.read_csv(path + 'AM_AL_spec_P100_20Per.csv') 
        v100_20per = pd.read_csv(path + 'AM_AL_spec_V100_20Per.csv') 
        p100_20per_d = p100_20per.drop(columns=['architecture','input','application_name','kernelname'])
        #print(list(p100_20per.columns.values))
  
        p100_20per_vals = p100_20per_d.values
        p100_20per_temp = p100_20per.drop(p100_20per.columns[0],axis=1,inplace=True)
        p100_20per_vals = p100_20per_temp.values.astype(float)
        ##p100_20per_scaler = StandardScaler().fit(p100_20per_vals)
        ##p100_20per_norm = p100_20per_scaler.transform(p100_20per_vals)
  
        v100_20per = v100_20per.drop(columns=['architecture','input','application_name','kernelname'])
        v100_20per_temp = v100_20per.drop(v100_20per.columns[0],axis=1,inplace=True)
        v100_20per_vals = v100_20per_temp.values.astype(float)
        #v100_20per_scaler = StandardScaler().fit(v100_20per_vals)
        ##v100_20per_norm = v100_20per_scaler.transform(v100_20per_vals)
  
        X_trainP, X_testP, y_trainV, y_testV = train_test_split(p100_20per_vals, v100_20per_vals, test_size=.33, random_state=42)
        X_trainP, X_testP, y_trainV, y_testV = X_trainP[:, 1:], X_testP[:, 1:], y_trainV[:, 1:], y_testV[:, 1:] 
  
        print(X_trainP.shape, y_trainV.shape, X_testP.shape, y_testV.shape)
        return (X_trainP, y_trainV), (X_testP, y_testV)        
    else:
        ## Creating load_data for deep hyper
        # Combining V100 and P100 on same row for same run
        # We are deleting cases where there is no run for either of the architectures
        # Every column name is appended with the name of the architecture (e.g. "_V100");
        # This includes the `master_index` (e.g `master_index_V100`)
        ##path = '/Users/yzamora/power/nvidia_gpus/all_apps/'
        path = '/home/yzamora/power_update/nvidia_gpus/all_apps/'
        df_joined = pd.read_csv(path + 'df_master_joined.csv')
        #df_joined = pd.read_parquet(path + 'df_master_joined.parquet')
        df_joined.master_index_P100 = df_joined.master_index_P100.astype('int64') # Make sure index is integer
        df_joined.master_index_V100 = df_joined.master_index_V100.astype('int64') # Make sure index is integer
        ##df_joined.shape

        # This is an "empty" dataframe (meaning no rows), containing
        # column names for numerical data only.
        # The column nmaes can be used to index the columns of the
        # scaled data (in master_scaled_data.npy)
        df_columns_only = pd.read_csv(path + 'df_column_reference.csv')
        #df_columns_only = pd.read_parquet(path + 'df_column_reference.parquet')
        ##df_columns_only


        # This is a 2-D numpy array corresponding to the numerical data in 'df_master.parquet'
        # The data has been scaled using the StandardScaler in scikitlearn

        # Notes:
        #   - The row indices correspond to the `master_index` column of 'df_master.parquet'
        #   - The columns correspond to the columns in 'df_column_reference.parquet'.
        #     (e.g. can use `df.get_loc(column-name)` to get the column index)

        master_data_scaled = np.load(path + 'master_scaled_data.npy')
        ##master_data_scaled.shape


        df = df_joined.copy()  # Start with all of df_joined

        # Target index and values
        target_index = df['master_index_V100'].values
        target = master_data_scaled[ target_index ]

        # Training data index and values
        data_index = df['master_index_P100'].values
        data = master_data_scaled[ data_index ]


        # Split the data for training
        (
            X_train, X_test,
            y_train, y_test,
        ) = train_test_split(
            data,
            target,
            random_state=42,
            test_size=.33
        )
 
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    load_data()
