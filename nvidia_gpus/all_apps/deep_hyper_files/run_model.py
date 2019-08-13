#!/usr/bin/env python
#Step 1
import os
import time
import json
import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from scipy import stats
#from pylab import rcParams
from sklearn.utils import check_random_state
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
#from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Importing random forest model and libraries
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
#from sklearn.model_selection import cross_val_score
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import regularizers
from keras import optimizers
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import scipy as sp
import pickle as pkl


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Creates array of ipc values and array of metrics
def process_keep(new_df):
    #new_def_norm = new_df.drop(columns=['shared_utilization','stall_other','single_precision_fu_utilization','architecture','input','ipc','application_name','kernelname'])
    new_def_norm = new_df.drop(columns=['architecture','input','application_name','kernelname'])

    #v100+p100 combined
    new_def_norm_values = new_def_norm.values
    ##new_def_norm = MinMaxScaler().fit_transform(new_def_norm_values)

    ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!####
    ##X_VP = new_def_norm

    # NOT SPLITTING HERE - Split the data up in train and test sets
    ##X_trainVP, X_testVP, y_trainVP, y_testVP = train_test_split(X_VP, Y_VP, test_size=test_size, random_state=42)

    # Import `StandardScaler` from `sklearn.preprocessing`
    from sklearn.preprocessing import StandardScaler

    # Define the scaler
    #scalerVP = StandardScaler().fit(X_trainVP)
    ##scalerVP = StandardScaler().fit(X_VP)
    scalerVP = StandardScaler().fit(new_def_norm_values)

    # Scale the train set
    ##X_trainVP = scalerVP.transform(X_VP)
    X_trainVP = scalerVP.transform(new_def_norm_values)

    # Scale the test set
    #X_testVP = scalerVP.transform(X_testVP)

    return X_trainVP


##weighted mse
import keras.backend as K
def weighted_mse(loss_weight):
    def loss(y_true, y_pred):
        #import pdb; pdb.set_trace()
        #y_true = K.variable(y_true)
        #y_pred = K.variable(y_pred)
        #loss = K.mean(K.dot(K.square(y_true - y_pred),loss_weight))
        loss = K.mean(K.square(y_true - y_pred)*loss_weight)
        return loss

    return loss

#ipc = 96, dram_read_throughput = 33,dram_write_throughput = 34,  ldst_fu_utilization=105
#cf_fu_utilization = 106,
#loss_weight = np.ones((116,1)) ##setting model to zero on those two items and see how it changes
loss_weight = np.zeros(116)
loss_weight[96] = 0#17
loss_weight[33] = 100#15 ## increasing dram write and read weights
loss_weight[34] = 100#14
loss_weight[105] = 0#13
loss_weight[106] = 0#10
#print(loss_weight)


#creating

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
#dataset = pd.read_csv("p100_only.csv", delimiter=",").values
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
# create model
def my_model(l2_weight,k_init = 'glorot_uniform',lr_num=.001):
    model = Sequential()
    """
    #original simple dl model
    model.add(Dense(12, input_dim=112, kernel_initializer='normal',activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='relu'))"""
    #early stopping, smaller layers, less layers

    model.add(Dense(130, input_dim=116, kernel_initializer=k_init,activation='relu', kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(125, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(125, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(120, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(120, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(120, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(120, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(120, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(120, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(125, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(125, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(125, activation='relu',kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight)))
    #model.add(Dense(90, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    #model.add(Dense(110, activation='relu'))
    #rerun and check everything is normalized correctly
    model.add(Dense(125, activation='relu',kernel_initializer=k_init, kernel_regularizer=regularizers.l2(l2_weight)))
    ##model.add(Dense(116, activation='relu')) #, kernel_regularizer=regularizers.l2(l2_weight)))
    model.add(Dense(116, kernel_initializer=k_init,kernel_regularizer=regularizers.l2(l2_weight),activation= None))
    # Compile model
    #mean absolute percentage error - indicating that we seek to minimize the mean percentage difference between
    #predicted ipc and the actual ipc
    #loss_weight = vector of weights
    #model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    #changing step size - halfing the step size
    adam_step = optimizers.Adam(lr=lr_num)
    model.compile(loss=weighted_mse(loss_weight), optimizer=adam_step, metrics=['mse','mae'])
    # Fit the model
    ## model.fit(X, Y, epochs=10, batch_size=10) ##works
    #look at weighted mean square error - putting more weight on certain metrics
    """
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))"""
    return model

#loading memory bound metrics
XV_val = np.load('../X_predicted_test_v100.npy')
XP_val = np.load('../X_predicted_test_p100.npy')

new_p = pd.read_csv('../train_predicted_true_p100.csv')

def get_w_vec(df, weights, indices=None):
    w = []
    indices = indices or [i for i in range(len(df))]
    for ind in indices:
        default = 1.0
        name = df.iloc[ind]["application_name"]
        w.append(weights.get(name, default))
    return w

train_weights = get_w_vec(
    new_p,
    {
        "backprop": 1, #-20.0,
        "stream": 1, #10000.0,
        "leuokocyte": 1, #10000.0,
        "hybridsort": 1, #10000.0,
        "kmeans": 100,
        "srad" :1
    },
)

## Training model with various training data sizes
def apply_std_scaler_(data_in):
    from sklearn.preprocessing import StandardScaler
    scaler_ = StandardScaler().fit(data_in)
    return scaler_.transform(data_in)

XP_val = apply_std_scaler_(XP_val)
XV_val = apply_std_scaler_(XV_val)

#step 4
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
all_data = False
p100_p100 = False
p100_v100 = True
def testing_training_size(activation, depth,nunits,regularization,k_init = 'glorot_uniform', lr = .001,EPOCHS = 9000, BATCH_SIZE = 50):
    #training size decreases
    if all_data:
        new_p = pd.read_csv('cor_p100.csv', index_col = 0)
        new_v = pd.read_csv('cor_v100.csv', index_col = 0)
    else:
        ##new_p = pd.read_csv('/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/with_same_base_P100/AM_AL_spec_P100_20per.csv', index_col = 0) #using specified kernels
        ##new_v = pd.read_csv('/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/with_same_base_P100/AM_AL_spec_V100_20per.csv', index_col = 0)
        new_p = pd.read_csv('../train_predicted_true_p100.csv') #using specified kernels
        new_v = pd.read_csv('../train_predicted_true_v100.csv')

        #X_trainP, X_testP, y_trainP, y_testP, X_P, Y_P = process(new_p,test_size)
        #X_trainV, X_testV, y_trainV, y_testV, X_V, Y_V = process(new_v,test_size)
        #import pdb; pdb.set_trace()
        #need to preprocess data with splitting it
        X_trainP = process_keep(new_p)
        X_trainV = process_keep(new_v)


        h_model = my_model(regularization,k_init,lr)
        #pass in validation data - remove validation split, do not split training set again
        #history = h_model.fit(X_trainP, X_trainV, epochs=100, batch_size=500,  verbose=1, validation_split=0.2)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
        ###mc = ModelCheckpoint('adam_step-noappweight_membounds2_dram_kmeans_zerobias.h5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)
        #h_model.save('wsame_PV_dl_10per_woutbias' + str(int(regularization)) +'.h5')
        #create a separate vector with a weight for the applications - put this vector into sample weight, after callbacks
        #sample_weight = just do regular train, test, split
        history = h_model.fit(X_trainP, X_trainV, epochs=EPOCHS, batch_size=BATCH_SIZE,  verbose=1, validation_data=(XP_val,XV_val),callbacks=[earlystop,],sample_weight=np.array(train_weights))
        #h_model.save('wAsame_PV_dl_10per_woutbias' + str(int(regularization)) +'.h5')
        lowest_val = np.min(history.history['val_loss'])
        return lowest_val



#Going through 10 different training set sizes and saving results to val_testsize
def run(param_dict):
    BATCH_SIZE = param_dict['batch_size']
    EPOCHS = param_dict['epochs']
    val_testsize = {}
    new_exp = -7.444444444444445
    mult = 2.8684210526315788
    val_testsize[2.86]= testing_training_size(param_dict['activation'], param_dict['depth'],param_dict['nunits'],mult*10**new_exp,lr=0.0001,EPOCHS=EPOCHS,BATCH_SIZE=BATCH_SIZE)
    print("lowest val loss", val_testsize[2.86])
if __name__=='__main__':
    from problem1 import Problem
    param_dict = Problem.starting_point_asdict
    param_dict['epochs'] = 10
    """param_dict={}
    param_dict['batch_size'] = 50
    param_dict['epochs'] = 500"""
    run(param_dict)
