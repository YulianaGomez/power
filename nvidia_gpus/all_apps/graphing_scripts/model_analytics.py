import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import sys
import glob
import os
import pickle

sys.path.append('/Users/yzamora/deephyper/deephyper/search/nas/model')
from train_utils import selectMetric
import scipy as sp

#selection type can be either Random or AL or C
#change unique to best or best to unique
selection_type = 'C'
if selection_type == 'AL':
    type_c = 'best'
else:
    type_c = 'unique'
val_summary_dict = {'Selection':[],'Percent':[],'Application':[],'MAPE':[]}
model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/deephyper_final_models/' + selection_type +'_one_20-POST_' + type_c + '.h5'
#model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/deephyper_models/best_model_18_per20.h5'

model = tf.keras.models.load_model(model_path,
    custom_objects={
        'r2': selectMetric('r2')
    }
                               )
# When using .sav models
###model = pickle.load(open(model_path,'rb'))

"""
X_test = np.load('/Users/yzamora/active_learning/X_val_RANDOM_20per.npy')
y_test = np.load('/Users/yzamora/active_learning/y_val_RANDOM_20per.npy')
"""
app_single_name = "leukocyte"
##X_test = np.load('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/Random_val_indices/Random_20per_' + app_single_name + '_x_val.npy')
##y_test = np.load('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/Random_val_indices/Random_20per_' + app_single_name + '_y_val.npy')
if selection_type == 'C':
    path = glob.glob('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/Random_val_indices/*')
else:
    path = glob.glob('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/' + selection_type + '_val_indices/*')

for f in path:
    xval_values = np.load(f)
    xval_name = os.path.basename(f)

    if 'x_val' in xval_name:
        #val_summary_dict['Selection'].append(xval_name.split('_')[0])
        val_summary_dict['Selection'].append(selection_type)
        val_summary_dict['Percent'].append(xval_name.split('_')[1].split('p')[0])
        val_summary_dict['Application'].append(xval_name.split('_')[2])
        yval_path = f.replace('x_val','y_val')
        yval_values = np.load(yval_path)
        ##print(xval_name,xval_name.replace('x_val','y_val'))
        #when running through all xval and yvals
        X_test = xval_values
        y_test = yval_values

        y_pred = model.predict(X_test)
        """
        print(model.metrics_names)
        print(model.evaluate(X_test,y_test))
        """
        #Averaging down the columns first (metrics), which converts it to 1 error per metric, then averages that (one row)
        mse = np.mean(np.mean(np.square(y_pred-y_test),axis=0))
        rmse = np.sqrt(mse)

        normX_testV = np.mean(np.mean(np.square(y_test),axis=0))
        #relative to the full scale of all the test data - 3x bigger because it's one number scaling across all metrics
        mse_relative = np.mean(np.mean(np.square(y_pred-y_test),axis=0))/normX_testV
        mspe = np.mean(np.mean(np.square(y_pred - y_test),axis=0))/normX_testV ## ask bethany
        rmse_relative = np.sqrt(mspe)

        """"""
        """
        print ("R:",  sp.stats.pearsonr(y_test.flatten(), y_pred.flatten())[0])
        print ("MAE:", np.mean(np.abs(y_pred.flatten() - y_test.flatten())))
        print ("RMSE:", np.sqrt(np.power(y_test.flatten()- y_pred.flatten(), 2).mean()))
        """
        #left out the mean, i would get the error rate for each metric
        #first show val error decreasings as training data increases
        #second show that val error can decreases earlier with active learning implementation

        #print("MSE:", mse)
        #print("RMSE:", rmse)
        #print("MSPE:", mspe)

        from sklearn.externals.joblib import load
        ipc_scaler = load('/Users/yzamora/power/nvidia_gpus/all_apps/scalers/std_scaler_ipc.bin')
        y_pred_unscaled = ipc_scaler.inverse_transform(y_pred)
        y_test_unscaled = ipc_scaler.inverse_transform(y_test)
        try:
            y_old = ipc_scaler.inverse_transform(X_test[:,96])
        except:
            import pdb; pdb.set_trace()
        #y_old = ipc_scaler.inverse_transform(X_test[96])


        def mean_absolute_percentage_error(y_true, y_pred):
            """y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100"""
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
            return np.mean( np.divide( np.abs( np.subtract( y_true, y_pred) ), y_true) ) * 100.0

        MAPE = mean_absolute_percentage_error(y_test_unscaled,y_pred_unscaled)
        old_new = mean_absolute_percentage_error(y_old,y_test_unscaled)
        ##print("MAPE for " + xval_name, MAPE)
        val_summary_dict['MAPE'].append(MAPE)
        ##val_summary_dict['MAPE'].append(old_new)
        #print("MAPE",MAPE)
        #print(val_summary_dict)

##df_val_summary = pd.DataFrame(val_summary_dict)
##df_val_summary.to_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/' + selection_type + '_val_MAPE_summary')
##tabbed twice

fig, ax = plt.subplots()

# Make the plot
ax.scatter(y_test, y_pred, alpha=0.5)

# Make it pretty
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_xlim())

ax.set_xlabel('True IPC')
ax.set_ylabel('Pred. IPC')
#ax.set_ylim([-10,30])
#ax.set_xlim([-10,30])
ax.axis('equal')
fig.set_size_inches(3.5, 3.5)
plt.title('Random 20 Percent: ' + app_single_name + ' (Scaled IPC Prediction)' + 'MAPE: ' + str(MAPE))
# Add in the goal line
##ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--');
plt.ylim([-2,2])
plt.plot(np.arange(-2,3),np.arange(-2,3))
plt.show()
