"""
Inter architecture prediction using datasets from large_dh_sets
November 22, 2019
"""
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

def my_model():
    model = Sequential()
    """
    #original simple dl model
    model.add(Dense(12, input_dim=112, kernel_initializer='normal',activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='relu'))"""
    #early stopping, smaller layers, less layers
    model.add(Dense(120, input_dim=116, kernel_initializer='normal',activation='relu'))
    model.add(Dense(110, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(90, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(110, activation='relu'))
    model.add(Dense(110, activation='relu'))
    model.add(Dense(1, activation=None))
    # Compile model
    #mean absolute percentage error - indicating that we seek to minimize the mean percentage difference between
    #predicted ipc and the actual ipc
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    # Fit the model
    ## model.fit(X, Y, epochs=10, batch_size=10) ##works

    """
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))"""
    return model

    ##testing largest file
X_train = np.load('/Users/yzamora/Desktop/X_train_fullset.npy')
y_train = np.load('/Users/yzamora/Desktop/y_train_fullset.npy')
X_val = np.load('/Users/yzamora/Desktop/X_val_fullset.npy')
y_val= np.load('/Users/yzamora/Desktop/y_val_fullset.npy')
model = my_model()
##XP_train, XP_val, yP_train, yP_val = train_test_split(X_test, y_test, test_size=.3, random_state=42)

#import pdb; pdb.set_trace()
model.fit(X_train, y_train, epochs=100, batch_size=250, verbose=1,validation_data=(X_val,y_val))
#import pdb; pdb.set_trace()
model_fname = "/Users/yzamora/Desktop/CL_fullset_model.h5"
model.save(model_fname)
"""
y_pred = model.predict(XP_val)

from sklearn.externals.joblib import load
x_scaler = load('/Users/yzamora/power/nvidia_gpus/all_apps/large_dh_sets/std_scaler_x.bin')
y_scaler = load('/Users/yzamora/Desktop/scalers_2/scaleripc.bin')
y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(yP_val)
"""

def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    """diffs = []
    for v in range(len(y_true)):
        #diffs.append(abs(max(y_pred[v], 0.0) - y_true[v]) / y_true[v])
        diffs.append(abs(y_pred[v] - y_true[v]) / y_true[v])
    return np.mean(diffs) * 100.0"""
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return np.mean( np.divide( np.abs( np.subtract( y_true, y_pred) ), y_true) ) * 100.0

    #for v in range(len(y_true)):
    #    y_pred[v] = max(y_pred[v], 0.0)
    #return np.mean( np.abs(y_pred - y_true) / y_true) * 100

"""
MAPE_score = mean_absolute_percentage_error(yP_val,y_pred)
print(MAPE_score)
# Plott
"""
