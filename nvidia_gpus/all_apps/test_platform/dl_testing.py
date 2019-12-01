##loading model - november 13, 2020
import numpy as np
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import r2_score
import tensorflow as tf
import keras
import glob
import os
import sys
import scipy as sp
import keras
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/yzamora/deephyper/deephyper/search/nas/model')
from train_utils import selectMetric

model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/deephyper_final_models/C_one_20-POST_unique.h5'
p_model = tf.keras.models.load_model(model_path,
            custom_objects={
                'r2': selectMetric('r2')
            }
        )


# Use when testing specific testing and training sets
"""path = '/Users/yzamora/active_learning/one_AL_indices/'
X_train = np.load(path + 'X_train_AL_20per_one.npy')
X_test = np.load(path + 'X_val_AL_20per_one.npy')
y_train = np.load(path + 'y_train_AL_20per_one.npy')
y_test = np.load(path + 'y_val_AL_20per_one.npy')"""

path = '/Users/yzamora/active_learning/large_dh_sets'
X_train = np.load(path + 'X_train.npy')
X_test = np.load(path + 'X_val.npy')
y_train = np.load(path + 'y_train.npy')
y_test = np.load(path + 'y_val.npy')

y_pred = p_model.predict(X_test)

from sklearn.externals.joblib import load
ipc_scaler = load('../scalers/std_scaler_ipc.bin')
y_pred = ipc_scaler.inverse_transform(y_pred)
y_test = ipc_scaler.inverse_transform(y_test)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100"""
    diffs = []
    for v in range(len(y_true)):
        #diffs.append(abs(max(y_pred[v], 0.0) - y_true[v]) / y_true[v])
        diffs.append(abs(y_pred[v] - y_true[v]) / y_true[v])
    return np.mean(diffs) * 100.0
#import pdb; pdb.set_trace()

# Plotting predicted vs true
#print ("R:",  sp.stats.pearsonr(X_testV.flatten(), y_pred.flatten())[0])
print ("MAE:", np.abs(y_pred - y_test).mean())
print ("RMSE:", np.sqrt(np.power(y_test- y_pred, 2).mean()))
print("MAPE:", mean_absolute_percentage_error(y_test,y_pred))

fig, ax = plt.subplots()

# Make the plot
ax.scatter(y_test, y_pred, alpha=0.5)

# Make it pretty
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_xlim())

ax.set_xlabel('True Metrics')
ax.set_ylabel('Pred. Metrics')

fig.set_size_inches(3.5, 3.5)

# Add in the goal line
ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--');
#plt.savefig("metric_predict.png")
