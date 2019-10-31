"""
Looking at conventional deep learning model to see separate evaluation scores
"""


import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
from pylab import rcParams
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Importing random forest model and libraries
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.model_selection import cross_val_score
import scipy as sp
import pickle as pkl

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense


"""
path = '/Users/yzamora/active_learning/'
X_train = np.load(path + 'X_train_AL_20per.npy')
X_test = np.load(path + 'X_val_AL_20per.npy')
y_train = np.load(path + 'y_train_AL_20per.npy')
y_test = np.load(path + 'y_val_AL_20per.npy')
"""
# Using same training data used in deephyper
path = '/Users/yzamora/active_learning/same_ALL_indices/'
X_train = np.load(path + 'X_train_30per_ALL.npy')
X_test = np.load(path + 'X_val_30per_ALL.npy')
y_train = np.load(path + 'y_train_30per_ALL.npy')
y_test = np.load(path + 'y_val_30per_ALL.npy')

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
#dataset = pd.read_csv("p100_only.csv", delimiter=",").values
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
# create model
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
    model.add(Dense(116, activation=None))
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

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
"""
#early stopping - once validation error stops improving, cut it off
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=100, verbose=1, mode='min')
callbacks=earlystop
p_model = my_model()
mc = ModelCheckpoint('filename.h5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)
#callbacks_list = [earlystop, mc]
#p_model.fit(X_trainP, X_trainV, epochs=2000, batch_size=10000, verbose=1,validation_split=0.7,callbacks=[earlystop,mc])
p_model.fit(X_train, y_train, epochs=500, batch_size=10000, verbose=1,validation_data=(X_train,y_train),callbacks=[earlystop,mc])
#validation_data=(X_train,y_train)
"""
model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/best_deephyper_mls/conventional_models/C_ALL_30-POST_unique.h5'
p_model = load_model(model_path)

y_pred = p_model.predict(X_test)
mse = np.mean(np.mean(np.square(y_pred-X_test),axis=0))

print('mse', mse)
print('r2 score', r2_score(X_test,y_pred))
R = sp.stats.pearsonr(X_test.flatten(), y_pred.flatten())[0]
print("R,", R)
