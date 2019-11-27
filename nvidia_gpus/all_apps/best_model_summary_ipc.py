import numpy as np
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import r2_score
import tensorflow as tf
import keras
import glob
import os
import sys
sys.path.append('/Users/yzamora/deephyper/deephyper/search/nas/model')
from train_utils import selectMetric
import scipy as sp
import pandas as pd
import pickle


def get_metrics(model_path, X_val_path, y_val_path, predict_column):
    if "RF_" in model_path:
        model = pickle.load(open(model_path,'rb'))
    else:
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

    #import pdb; pdb.set_trace()
    #Averaging down the columns first (metrics), which converts it to 1 error per metric, then averages that (one row)
    mse = np.mean(np.mean(np.square(y_pred-y_test),axis=0))
    rmse = np.sqrt(mse)
    #load scaler
    from sklearn.externals.joblib import load
    ipc_scaler = load('scalers/std_scaler_ipc.bin')
    y_pred = ipc_scaler.inverse_transform(y_pred)
    y_test = ipc_scaler.inverse_transform(y_test)

    def mean_absolute_percentage_error(y_true, y_pred):
        """y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100"""
        diffs = []
        for v in range(len(y_true)):
            #diffs.append(abs(max(y_pred[v], 0.0) - y_true[v]) / y_true[v])
            diffs.append(abs(y_pred[v] - y_true[v]) / y_true[v])
        return np.mean(diffs) * 100.0, np.std(diffs) * 100.0

    normX_testV = np.mean(np.mean(np.square(y_test),axis=0))
    #relative to the full scale of all the test data - 3x bigger because it's one number scaling across all metrics
    mse_relative = np.mean(np.mean(np.square(y_pred-y_test),axis=0))/normX_testV
    mspe = np.mean(np.mean(np.square(y_pred - y_test),axis=0))/normX_testV ## ask bethany
    rmse_relative = np.sqrt(mspe)

    R = sp.stats.pearsonr(y_test.flatten(), y_pred.flatten())[0]
    MAE = np.mean(np.abs(y_pred.flatten() - y_test.flatten()))
    RMSE = np.sqrt(np.power(y_test.flatten()- y_pred.flatten(), 2).mean())
    R2 = r2_score(y_test,y_pred)
    MAPE, MAPE_std = mean_absolute_percentage_error(y_test,y_pred)
    """"""
    print ("R:", R)
    print ("MAE:", MAE)
    print ("RMSE:", RMSE)
    print("R2:", R2)
    print("MAPE:", MAPE)
    print("MAPE-std:", MAPE_std)
    return R, MAE, RMSE, R2, MAPE, MAPE_std


root_model_dir = "/Users/yzamora/power/nvidia_gpus/all_apps/deephyper_final_models"
root_data_dir = "/Users/yzamora/active_learning"
output_csv_path = "/Users/yzamora/active_learning/dataframe_summary_ipc_std.csv"

do_summary = True
do_plot = True

if do_summary:
    # stype == selection type (AL, RANDOM, C)
    # mtype == metric type (one, ALL, RM)
    summary = {
        "stype": [],
        "mtype": [],
        "percent": [],
        "R": [],
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "MAPE":[],
        "USE_IPC":[],
        "MAPE_std":[],
    }

    #for loop, model_path in enumerate(glob.glob(root_model_dir + "/*.h5")):
    for loop, model_path in enumerate(glob.glob(root_model_dir + "/*.h5")):

        stype, mtype, percent = os.path.basename(model_path).split("_")[:3]
        percent = int(percent.split("-")[0])
        stype_use = stype
        if stype=='C' or stype=='RF': stype_use = 'RANDOM'
        # Get appropriate validation data
        sub_dir = root_data_dir + "/" + mtype + "_" + stype_use + "_indices"

        if not os.path.exists(sub_dir):
            print("WARNING: Skipping missing:", sub_dir)
            continue

        X_val_path = sub_dir + "/X_val_" + stype_use + "_" + str(percent) + "per_" + mtype + ".npy"
        y_val_path = sub_dir + "/y_val_" + stype_use + "_" + str(percent) + "per_" + mtype + ".npy"

        for USE_IPC in [0,1]:

            if USE_IPC:
                if mtype == "RM":
                    col = 0
                elif mtype == "ALL":
                    col = 96
                else:
                    col = None
            else:
                col = None
            R, MAE, RMSE, R2, MAPE, MAPE_std = get_metrics(model_path, X_val_path, y_val_path, col)

            # Update Summary Dict
            summary["stype"].append(stype)
            summary["mtype"].append(mtype)
            summary["percent"].append(percent)
            summary["R"].append(R)
            summary["MAE"].append(MAE)
            summary["RMSE"].append(RMSE)
            summary["R2"].append(R2)
            summary["MAPE"].append(MAPE)
            summary["MAPE_std"].append(MAPE_std)
            summary["USE_IPC"].append(USE_IPC)

        print("Loop", loop, "done.")
        ##if loop == 3: break;
    df = pd.DataFrame(summary)
    df.to_csv(output_csv_path, index=False)
    print(df)

if do_plot:
    import matplotlib.pyplot as plt

    df = pd.read_csv(output_csv_path)
    print(df)
    use_ipc = 1
    use_mtype = 'one'
    df = df[df["mtype"]== use_mtype]
    df = df[df["USE_IPC"]== use_ipc]
    df = df.set_index("percent").sort_index()

    groups = df.groupby(["stype", "mtype"])

    f = [ plt.figure() for i in range(5) ]
    ax = [ f_i.add_subplot(1, 1, 1) for f_i in f ]

    ax[0].set_prop_cycle(color=['red','green', 'blue', 'coral', 'black','magenta','cyan'],marker = ['.','^','p','v','o','o','s'])
    ax[1].set_prop_cycle(color=['red','green', 'blue', 'coral', 'black','magenta','cyan'],marker = ['.','^','p','v','o','o','s'])
    ax[2].set_prop_cycle(color=['red','green', 'blue', 'coral', 'black','magenta','cyan'],marker = ['.','^','p','v','o','o','s'])
    ax[3].set_prop_cycle(color=['red','green', 'blue', 'coral', 'black','magenta','cyan'],marker = ['.','^','p','v','o','o','s'])
    ax[4].set_prop_cycle(color=['red','green', 'blue', 'coral', 'black','magenta','cyan'],marker = ['.','^','p','v','o','o','s'])

    for name, group in groups:
        #import pdb; pdb.set_trace()
        dfg = group.reset_index()
        ax[0].plot(dfg["percent"], group["RMSE"], label=name)
        ax[1].plot(dfg["percent"], group["MAE"], label=name)
        ax[2].plot(dfg["percent"], group["R"], label=name)
        ax[3].plot(dfg["percent"], group["R2"], label=name)
        ax[4].plot(dfg["percent"], group["MAPE"], label=name)


    ax[0].set_title("RMSE (IPC only: " + str(bool(use_ipc)) + ")")
    ax[0].set_ylabel("RMSE")
    ax[0].set_xlabel("Percent")
    ax[0].legend(loc="best")

    ax[1].set_title("MAE (IPC only: " + str(bool(use_ipc)) + ")")
    ax[1].set_ylabel("MAE")
    ax[1].set_xlabel("Percent")
    ax[1].legend(loc="best")

    ax[2].set_title("R (IPC only: " + str(bool(use_ipc)) + ")")
    ax[2].set_ylabel("R")
    ax[2].set_xlabel("Percent")
    ax[2].legend(loc="best")

    ax[3].set_title("R2 (IPC only: " + str(bool(use_ipc)) + ")")
    ax[3].set_ylabel("R2")
    ax[3].set_xlabel("Percent")
    ax[3].legend(loc="best")

    #ax[4].set_title("MAPE (IPC only: " + str(bool(use_ipc)) + ")")
    ax[4].set_title("MAPE of P100 to V100 IPC predictions")
    ax[4].set_ylabel("MAPE")
    ax[4].set_xlabel("Percent")
    ax[4].legend(loc="best")
    ax[4].set_ylim([0,30])



    plt.show()
