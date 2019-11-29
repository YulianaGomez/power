import pickle
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import numpy as np

randomforest_path = '/Users/yzamora/Desktop/AL_new_models/RF_fullset.sav'
conventionaldl_path = '/Users/yzamora/Desktop/CL_fullset_model.h5'

rf_model = pickle.load(open(randomforest_path,'rb'))
cl_model = tf.keras.models.load_model(conventionaldl_path)

X_val = np.load('/Users/yzamora/Desktop/X_val_fullset.npy')
y_val = np.load('/Users/yzamora/Desktop/y_val_fullset.npy')

rf_pred = rf_model.predict(X_val)
cl_pred = cl_model.predict(X_val)

from sklearn.externals.joblib import load
ipc_scaler = load('/Users/yzamora/power/nvidia_gpus/all_apps/large_training_scalers/std_scaler_ipc.bin')
rf_pred_unscaled = ipc_scaler.inverse_transform(rf_pred)
cl_pred_unscaled = ipc_scaler.inverse_transform(cl_pred)
y_val = ipc_scaler.inverse_transform(y_val)

def mean_absolute_percentage_error(y_true, y_pred):
            """y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100"""
            diffs = []
            for v in range(len(y_true)):
                #diffs.append(abs(max(y_pred[v], 0.0) - y_true[v]) / y_true[v])
                diffs.append(abs(y_pred[v] - y_true[v]) / y_true[v])

            standard_dev = np.std(diffs)
            count = len(diffs)
            standard_error = standard_dev/np.sqrt(count)
            return np.mean(diffs) * 100.0, standard_error*100.0, count

print(mean_absolute_percentage_error(y_val,rf_pred_unscaled))
print(mean_absolute_percentage_error(y_val,cl_pred_unscaled))
