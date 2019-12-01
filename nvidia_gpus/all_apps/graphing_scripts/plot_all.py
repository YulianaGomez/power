import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append('/Users/yzamora/deephyper/deephyper/search/nas/model')
from train_utils import selectMetric
import scipy as sp


# Read in master CSV file
df = pd.read_csv('../all_data.csv', index_col = 0)

# Drop columns with NaN
df = df.dropna(axis=1,how='any')
df.reset_index(drop=True, inplace=True)

# Define peak memory bandwidth p100 732
peak_mem_bw = {
    "V100": 898.048 * (1024*1024*1024),
    "P100": 749.0 * (1024*1024*1024),
}
mem_bw_thresh = 0.75

# Add a column specifying if the case is memory bound
df_archs = []
for arch in peak_mem_bw.keys():
    df_tmp = df[df["architecture"] == arch].copy()
    new_col = (
        df_tmp["dram_read_throughput"] + df_tmp["dram_write_throughput"]
    ) / peak_mem_bw[arch]
    new_col = new_col > mem_bw_thresh
    df_tmp["memory_bound"] = new_col
    df_archs.append(df_tmp.copy())
df_merged = pd.concat(df_archs).sort_index()

stream_temp = df_merged[df_merged["application_name"] == 'stream']
# Convert bool "memory_bound" column to integers
df_merged["memory_bound"]= df_merged["memory_bound"].astype('int')

# Here we have our master dataframe (df_merged).
# Assume the numerical data from this dataframe is used to
# scale everything (also leave out `memory_bound` column).

# Helper funciton to return non-numerical column list
def _get_string_cols(df_in, str_cols=None):
    # Automatically detect non numerical columns
    str_cols = str_cols or []
    for col in df_in:
        if df_in[col].dtype == 'object':
            str_cols.append(col)
    return str_cols

# Convert numerical columns to out training/test
drop_cols = _get_string_cols(df_merged, ['memory_bound'])
data_to_scale = df_merged.drop(drop_cols, axis=1).values
scaler = StandardScaler().fit(data_to_scale)
scaled_data_ = scaler.transform(data_to_scale)

# Add column to df_merged called 'master_index'
df_merged['master_index'] = [int(i) for i in range(len(df_merged.index))]

# Lets create a dataframe (df_joined) with each row
# corresponding to a specific type of run.
# The V100 and P100 metrics are included in the same row,
# with `_V100` appended to the metric label for V100, etc.
# This means we have 2x the number of metrics for each row.
base = 'V100'
other = 'P100'

df_all = df_merged.copy()
unique_col = [] #column that has matches of kernels run on both architectures
for k, i in zip(df_all['kernelname'].values, df_all['input'].values):
    unique_col.append(k+'_'+i)
df_all['unique_index'] = unique_col
df_all.set_index('unique_index', inplace=True)

# Create 'base' and 'other' dataframes for join
df_b = df_all[df_all['architecture'] == base].copy()
df_o = df_all[df_all['architecture'] == other].copy()

# Final join operation, and drop rows with NaN elements
df_joined = df_b.join(df_o, lsuffix='_'+base, rsuffix='_'+other)
df_joined = df_joined.dropna(axis=0,how='any')
print("Number of runs that are both on P100 and V100: ", df_joined.shape)
##df_joined.head()

# We know know how to map dataframe columns to scaled-data columns
# However, we need to map column indices as well...
# Drop columns from df_merged to get a dataframe for
# determining the index of columns
drop_cols = _get_string_cols(df_merged, ['memory_bound','master_index'])
df_col_ref = df_merged.drop(drop_cols, axis=1).copy()

# We are assuming here that the columns of `df_col_ref`
# are ordered in the same way as `scaled_data_`

metric_basis = ['dram_read_throughput', 'dram_write_throughput', 'ipc']


# Now, lets see how predictions for V100 (given P100) compare...

#from deephyper.search.nas.model.train_utils import selectMetric
reg_model = False
if reg_model:
    def weighted_mse(loss_weight):
        def loss(y_true, y_pred):
            loss = K.mean(K.square(y_true - y_pred)*loss_weight)
            return loss
        return loss

    loss_weight = np.ones(116)
    loss_weight[96] = 1#17 ipc
    loss_weight[33] = 100#15 dram_read
    loss_weight[34] = 100#14 dram_Write
    loss_weight[105] = 1#13
    loss_weight[106] = 1#10
    #model = load_model('bslh_DL_wbias.h5',custom_objects={'loss': weighted_mse(loss_weight)}) # first attempt
    #model = load_model('20perdata_dram_bias.h5',custom_objects={'loss':weighted_mse(loss_weight)})
    model = load_model('../12deeper-noappweight_membounds2_dram_kmeans_zerobias.h5',custom_objects={'loss':weighted_mse(loss_weight)})
    ##testing deephyper returned model
else:
    ##model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/best_deephyper_mls/AL_one_20-POST_best.h5'
    model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/deephyper_final_models/AL_one_20-POST_best.h5'
    model = tf.keras.models.load_model(model_path,
        custom_objects={
            'r2': selectMetric('r2')
        }
                                      )
#model.summary()
df_plot = df_joined[df_joined['memory_bound_V100'] == 1].copy()

colors = ("black", "red", "green", "blue" ,"pink")
groups = ('backprop', 'hybridsort', 'kmeans', 'srad')

##df_plot = df_joined[df_joined['memory_bound_V100'] == 1].copy()
df_plot = df_joined.copy()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
import pdb
for color, group in zip(colors, groups):
    df_predict = df_plot[df_plot['application_name_V100'] == group].copy()
    indices_to_predict = [int(i) for i in df_predict['master_index_P100'].values]
    prediction = model.predict(scaled_data_[indices_to_predict])
    #pdb.set_trace()
    try:
        print(prediction.max())
    except:
        pass

    # Print measured V100 metrics
    indices = [int(i) for i in df_predict['master_index_V100'].values]
    x_col_ind = df_col_ref.columns.get_loc(metric_basis[0])
    y_col_ind = df_col_ref.columns.get_loc(metric_basis[1])
    x = scaled_data_[indices,x_col_ind]
    y = scaled_data_[indices,y_col_ind]
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=20, marker='o', label=group)

    # Print measured P100 metrics
    indices = [int(i) for i in df_predict['master_index_P100'].values]
    x_col_ind = df_col_ref.columns.get_loc(metric_basis[0])
    y_col_ind = df_col_ref.columns.get_loc(metric_basis[1])
    x = scaled_data_[indices,x_col_ind]
    y = scaled_data_[indices,y_col_ind]
    ax.scatter(x, y, alpha=0.8, c='none', edgecolors=color, s=100, marker='s')

    # Print V100 predictions (from measured P100 metrics)
    if indices_to_predict: # Need to check that there are predictions to plot here...
        #pdb.set_trace()
        x_col_ind = df_col_ref.columns.get_loc(metric_basis[0])
        y_col_ind = df_col_ref.columns.get_loc(metric_basis[1])
        x_pred = prediction[:,x_col_ind]
        y_pred = prediction[:,y_col_ind]
        #pdb.set_trace()
        #print ("R:",  sp.stats.pearsonr(scaled_data_.flatten(), prediction.flatten())[0])

        ax.scatter(x_pred, y_pred, alpha=0.8, c='none', edgecolors=color, s=400, marker='^')


plt.xlabel(metric_basis[0]+' [Scaled]')
plt.ylabel(metric_basis[1]+' [Scaled]')
plt.title('All data on V100\n(Squares == P100 Data, Triangles == Predicted V100 Data)')
plt.legend(loc='best')
plt.show()

x_col_ind = df_col_ref.columns.get_loc(metric_basis[0])
y_col_ind = df_col_ref.columns.get_loc(metric_basis[1])
x = prediction[:,x_col_ind]
y = prediction[:,y_col_ind]
print ("R:",  sp.stats.pearsonr(x.flatten(), y.flatten ())[0])
