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

#selection type can be either 'Random' or 'AL' or 'C', 'OldNew', 'RF', 'RFAL'
#change unique to best or best to unique
selection_type_inputs = ['AL', 'Random', 'C', 'OldNew', 'RF', 'RFAL']
app_single_name = "stream"
single_run = False

for selection_type in selection_type_inputs:
    if selection_type == 'C':
        type_c = 'unique'
    else:
        type_c = 'best'
    val_summary_dict = {'Selection':[],'Percent':[],'Application':[],'MAPE':[], 'MAPE-std':[], 'count':[], 'delta_error':[]}
    model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/deephyper_final_models/' + selection_type +'_one_20-POST_' + type_c
    #model_path = '/Users/yzamora/power/nvidia_gpus/all_apps/deephyper_models/best_model_18_per20.h5'

    # When using .sav models
    model = None
    if selection_type in ["RF", "RFAL"]:
        model = pickle.load(open(model_path + '.sav','rb'))
    elif selection_type != 'OldNew':
        model = tf.keras.models.load_model(model_path + '.h5',
            custom_objects={
                'r2': selectMetric('r2')
            }
                                       )

    if selection_type in ['C', 'OldNew', 'RF']:
        path = glob.glob('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/Random_val_indices/Random' + '*.npy')
    elif selection_type == 'RFAL':
        path = glob.glob('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/AL_val_indices/AL*.npy')
    else:
        path = glob.glob('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/' + selection_type + '_val_indices/'+ selection_type + '*.npy')


    for f in path:
        xval_name = os.path.basename(f)

        if single_run:
            X_test = np.load('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/' + selection_type + '_val_indices/' + selection_type + '_20per_' + app_single_name + '_x_val.npy')
            y_test = np.load('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/' + selection_type + '_val_indices/' + selection_type + '_20per_' + app_single_name + '_y_val.npy')
            y_pred = model.predict(X_test)

        elif 'x_val' in xval_name:
            xval_values = np.load(f)
            #val_summary_dict['Selection'].append(xval_name.split('_')[0])
            val_summary_dict['Selection'].append(selection_type)
            val_summary_dict['Percent'].append(xval_name.split('_')[1].split('p')[0])
            val_summary_dict['Application'].append(xval_name.split('_')[2])
            yval_path = f.replace('x_val','y_val')
            yval_values = np.load(yval_path)
            ##print(xval_name,xval_name.replace('x_val','y_val'))
            #when running through all xval and yvals"""
            ## two tabs
            X_test = xval_values
            y_test = yval_values
            if model:
                y_pred = model.predict(X_test)

        else:
            continue

        """
        print(model.metrics_names)
        print(model.evaluate(X_test,y_test))
        """
        if model:
            #Averaging down the columns first (metrics), which converts it to 1 error per metric, then averages that (one row)
            mse = np.mean(np.mean(np.square(y_pred-y_test),axis=0))
            rmse = np.sqrt(mse)

            normX_testV = np.mean(np.mean(np.square(y_test),axis=0))
            #relative to the full scale of all the test data - 3x bigger because it's one number scaling across all metrics
            mse_relative = np.mean(np.mean(np.square(y_pred-y_test),axis=0))/normX_testV
            mspe = np.mean(np.mean(np.square(y_pred - y_test),axis=0))/normX_testV ## ask bethany
            rmse_relative = np.sqrt(mspe)


        from sklearn.externals.joblib import load
        ipc_scaler = load('/Users/yzamora/power/nvidia_gpus/all_apps/scalers/std_scaler_ipc.bin')
        if model:
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
            diffs = []
            for v in range(len(y_true)):
                #diffs.append(abs(max(y_pred[v], 0.0) - y_true[v]) / y_true[v])
                diffs.append(abs(y_pred[v] - y_true[v]) / y_true[v])

            standard_dev = np.std(diffs)
            count = len(diffs)
            standard_error = standard_dev/np.sqrt(count)
            return np.mean(diffs) * 100.0, standard_error*100.0, count

        def diff_error(p100_t, v100_t, v100_p):
            diffs = []
            for v in range(len(p100_t)):
                # finding difference between true values
                delta_t = v100_t[v] - p100_t[v]
                # finding difference between predicted ipc and old ipc
                delta_p = v100_p[v] - p100_t[v]
                #only looking at predictions where there is a signficant IPC difference
                if abs(delta_t) > 0.1:
                    diffs.append((delta_p - delta_t) / abs(delta_t))
                #true difference
                #diffs.append(delta_t)
            return np.mean(diffs) #* 100.0

        if selection_type == 'OldNew':
            MAPE, MAPE_std, counta = mean_absolute_percentage_error(y_old,y_test_unscaled)
            delta_error = diff_error(y_old, y_test_unscaled, y_old)
        else:
            MAPE, MAPE_std, counta = mean_absolute_percentage_error(y_test_unscaled,y_pred_unscaled)
            delta_error = diff_error(y_old, y_test_unscaled, y_pred_unscaled)

        #import pdb; pdb.set_trace()
        ##print("MAPE for " + xval_name, MAPE)
        val_summary_dict['MAPE'].append(MAPE)
        val_summary_dict['MAPE-std'].append(MAPE_std)
        val_summary_dict['count'].append(counta)
        val_summary_dict['delta_error'].append(delta_error)

        if single_run:
            break

        ##val_summary_dict['MAPE'].append(old_new)
        #print("MAPE",MAPE)
        #print(val_summary_dict)

    df_val_summary = pd.DataFrame(val_summary_dict)
    df_val_summary.to_csv('/Users/yzamora/power/nvidia_gpus/all_apps/specified_application_indices/' + selection_type + '_val_MAPE_error_summary')
    ##tabbed twice

"""
fig, ax = plt.subplots()

# Make the plot
ax.scatter(y_test_unscaled, y_pred_unscaled, alpha=0.5)

# Make it pretty
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_xlim())

ax.set_xlabel('True IPC')
ax.set_ylabel('Predicted IPC')
ax.set_ylim([0.1,.5])
ax.set_xlim([0.1,.5])
ax.axis('equal')
fig.set_size_inches(6, 6)
plt.title(selection_type + ' 20 Percent: ' + app_single_name + ' (Unscaled IPC Prediction)' + 'MAPE: ' + str(MAPE))
##plt.title(selection_type + ' 20 Percent: ' + xval_name.split('_')[2] + ' Unscaled IPC prediction) ' + 'MAPE: ' +str(MAPE))
## Add in the goal line
#ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--');
#plt.ylim([0,1])
plt.plot(np.arange(0,.6,.1),np.arange(0,.6,.1))
plt.show()
"""
