import pandas as pd
import warnings
import numpy as np
import glob
import os
import sys
from sklearn.preprocessing import StandardScaler
#
#warnings.filterwarnings("ignore")

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

#reading in data point indices from active learner
##p100_activeLearn_points = pd.read_csv('AL_Var_indices_size_250_10per.txt', names=["vals"]).iloc[:,0]
##path = '/Users/yzamora/power/nvidia_gpus/all_apps/mem_bound/'
"""
path = '/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/AL_var_indices/'
p100_activeLearn_points = pd.read_csv(path + 'AL_Var_indices_size_250_10per.txt', names=["vals"]).iloc[:,0]"""
##p100_activeLearn_points = pd.read_csv(path + 'Random_Var_indices_size_250_20per.txt', names=["vals"]).iloc[:,0]

#either 'AL_indices' or 'random_indices' path to indices created
#(txt files with unique identifiers)
selection_txt = 'AL_Var_indices'
path = '/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/' + selection_txt + '/'
path_all = glob.glob('/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/' + selection_txt + '/*')

##path = '/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/random_indices/'
"""path = '/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/AL_var_indices/'
p100_activeLearn_points = pd.read_csv(path + 'AL_Var_indices_size_250_20per.txt', names=["vals"]).iloc[:,0]"""


for f in path_all:
    #if True:
    indice_dir_name = os.path.basename(f)
    print(indice_dir_name)
    p100_activeLearn_points = pd.read_csv(path + indice_dir_name, names=["vals"]).iloc[:,0]
    ##indice_dir_name = 'AL_Var_indices_size_250_20per.txt'
    v100_activeLearn_points = p100_activeLearn_points.tolist()
    #creating a list of all strings
    v100_activeLearn_points = [val.replace("P100", "V100") for val in v100_activeLearn_points]
    #pd.series passing a list to the series constructor to create a pandas series
    v100_activeLearn_points = pd.Series(v100_activeLearn_points)

    #Creating pandas dataframe with the specified unique indices
    df_master_p100_activeLearn = df_master_p100.loc[p100_activeLearn_points]
    df_master_v100_activeLearn = df_master_v100.loc[v100_activeLearn_points]
    ##print('df_master_p100_activeLearn', df_master_p100_activeLearn)
    ##sys.exit(0)
    df_master_p100_NOT_activeLearn = df_master_p100[~df_master_p100.index.isin(p100_activeLearn_points.tolist())]
    df_master_v100_NOT_activeLearn = df_master_v100[~df_master_v100.index.isin(v100_activeLearn_points.tolist())]

    df_tmp = df_master_p100_NOT_activeLearn.dropna(axis=0,how='any')

    from sklearn.model_selection import train_test_split

    ##Need to created validation and test set from points NOT in active learner
    p100_drop_col = df_master_p100_NOT_activeLearn.drop(columns=['memory_bound','master_index','architecture','input','application_name','kernelname'])
    p100_data_pts = p100_drop_col.values

    v100_drop_col = df_master_v100_NOT_activeLearn.drop(columns=['memory_bound','master_index','architecture','input','application_name','kernelname'])
    v100_drop_col = df_master_v100_NOT_activeLearn.drop(columns=['memory_bound','master_index','architecture','input','application_name','kernelname'])
    v100_data_pts = v100_drop_col.values

    X_val, X_test, y_val, y_test  = train_test_split(p100_data_pts,v100_data_pts,test_size=.50, random_state=42)


    # Generate the same split with the indices as well...
    index_values_p100 = df_master_p100_NOT_activeLearn.index.values
    index_values_v100 = df_master_v100_NOT_activeLearn.index.values
    (
        val_p100, test_p100,
        val_v100, test_v100,
        val_p100_ind, test_p100_ind,
        val_v100_ind, test_v100_ind
    ) = train_test_split(
        p100_data_pts,
        v100_data_pts,
        index_values_p100,
        index_values_v100,
        test_size=.50,
        random_state=42
    )

    # Nice.. Looks like our old and new splits agree:
    assert (X_val == val_p100).all()
    assert (y_val == val_v100).all()
    assert (X_test == test_p100).all()
    assert (y_test == test_v100).all()

    # We can simplify the naming a bit to agree with the other naming:
    X_val_indices = val_p100_ind
    y_val_indices = val_v100_ind
    X_test_indices = test_p100_ind
    y_test_indices = test_v100_ind

    # How to get a kernel-specific part of the validation set:
    X_validation_df = df_master_p100_NOT_activeLearn.loc[pd.Series(X_val_indices)]
    y_validation_df = df_master_v100_NOT_activeLearn.loc[pd.Series(y_val_indices)]

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
    ##print(indice_dir_name)
    #import pdb; pdb.set_trace()

    uniq_app_id = indice_dir_name.split('_')[0]
    per_app = indice_dir_name.split('_')[-1].split('.')[0]
    for apps in groups:
        print(X_validation_sets[apps].shape, y_validation_sets[apps].shape)

        #print((np.reshape(y_validation_sets[apps],(y_validation_sets[apps].shape[0],1))).shape)
        print('specified_application_indices/AL_var_val/'+ uniq_app_id + '_' + per_app + '_' + apps + '_x_val')
        np.save('specified_application_indices/AL_var_val/'+ uniq_app_id + '_' + per_app + '_' + apps + '_x_val',X_validation_sets[apps])
        np.save('specified_application_indices/AL_var_val/'+ uniq_app_id + '_' + per_app + "_" + apps + '_y_val',np.reshape(y_validation_sets[apps],(y_validation_sets[apps].shape[0],1)))
