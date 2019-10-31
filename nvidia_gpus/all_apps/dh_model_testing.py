import numpy as np
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import r2_score
import tensorflow as tf
import glob
import os
import sys
sys.path.append('/Users/yzamora/deephyper/deephyper/search/nas/model')
from train_utils import selectMetric
import scipy as sp
import pandas as pd

model_path = "/Users/yzamora/power/nvidia_gpus/all_apps/best_deephyper_mls/AL_ALL_10-POST_best.h5"
X_val_path = "/Users/yzamora/active_learning/ALL_AL_indices/X_val_AL_10per_ALL.npy"
y_val_path = "/Users/yzamora/active_learning/ALL_AL_indices/y_val_AL_10per_ALL.npy"

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_metrics(model_path, X_val_path, y_val_path, predict_column):
    model = tf.keras.models.load_model(model_path,
        custom_objects={
            'r2': selectMetric('r2')
        }
    )
    X_test = np.load(X_val_path)
    y_test = np.load(y_val_path)

    y_pred = model.predict(X_test)
    if predict_column:
        y_pred = y_pred[:,predict_column]
        y_test = y_test[:,predict_column]

    #Averaging down the columns first (metrics), which converts it to 1 error per metric, then averages that (one row)
    mse = np.mean(np.mean(np.square(y_pred-y_test),axis=0))
    rmse = np.sqrt(mse)

    normX_testV = np.mean(np.mean(np.square(y_test),axis=0))
    #relative to the full scale of all the test data - 3x bigger because it's one number scaling across all metrics
    mse_relative = np.mean(np.mean(np.square(y_pred-y_test),axis=0))/normX_testV
    mspe = np.mean(np.mean(np.square(y_pred - y_test),axis=0))/normX_testV ## ask bethany
    rmse_relative = np.sqrt(mspe)

    R = sp.stats.pearsonr(y_test.flatten(), y_pred.flatten())[0]
    MAE = np.mean(np.abs(y_pred.flatten() - y_test.flatten()))
    RMSE = np.sqrt(np.power(y_test.flatten()- y_pred.flatten(), 2).mean())
    R2 = r2_score(y_test,y_pred)
    #import pdb; pdb.set_trace()
    """"""
    print ("R:", R)
    print ("MAE:", MAE)
    print ("RMSE:", RMSE)
    print("R2:", R2)
    print("MSE", mse)
    return R, MAE, RMSE, R2

get_metrics(model_path,X_val_path,y_val_path,0)
