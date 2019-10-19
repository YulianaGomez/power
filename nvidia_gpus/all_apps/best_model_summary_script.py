import numpy as np
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import glob
import os
import sys
sys.path.append('/Users/yzamora/deephyper/deephyper/search/nas/model')
from train_utils import selectMetric
import scipy as sp
import pandas as pd


def get_metrics(model_path, X_val_path, y_val_path):
    model = tf.keras.models.load_model(model_path,
        custom_objects={
            'r2': selectMetric('r2')
        }
    )
    X_test = np.load(X_val_path)
    y_test = np.load(y_val_path)

    y_pred = model.predict(X_test)

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

    """"""
    print ("R:", R)
    print ("MAE:", MAE)
    print ("RMSE:", RMSE)

    return R, MAE, RMSE


root_model_dir = "/Users/yzamora/power/nvidia_gpus/all_apps/best_deephyper_mls"
root_data_dir = "/Users/yzamora/active_learning"
output_csv_path = "/Users/yzamora/active_learning/dataframe_summary.csv"

do_summary = False
do_plot = True

if do_summary:
    # stype == selection type (AL, RANDOM)
    # mtype == metric type (one, ALL, RM)
    summary = {
        "stype": [],
        "mtype": [],
        "percent": [],
        "R": [],
        "MAE": [],
        "RMSE": [],
    }

    for loop, model_path in enumerate(glob.glob(root_model_dir + "/*.h5")):
        stype, mtype, percent = os.path.basename(model_path).split("_")[:3]
        percent = int(percent.split("-")[0])

        # Get appropriate validation data
        sub_dir = root_data_dir + "/" + mtype + "_" + stype + "_indices"

        if not os.path.exists(sub_dir):
            print("WARNING: Skipping missing:", sub_dir)
            continue

        X_val_path = sub_dir + "/X_val_" + stype + "_" + str(percent) + "per_" + mtype + ".npy"
        y_val_path = sub_dir + "/y_val_" + stype + "_" + str(percent) + "per_" + mtype + ".npy"
        R, MAE, RMSE = get_metrics(model_path, X_val_path, y_val_path)

        # Update Summary Dict
        summary["stype"].append(stype)
        summary["mtype"].append(mtype)
        summary["percent"].append(percent)
        summary["R"].append(R)
        summary["MAE"].append(MAE)
        summary["RMSE"].append(RMSE)

        print("Loop", loop, "done.")

    df = pd.DataFrame(summary)
    df.to_csv(output_csv_path, index=False)
    print(df)

if do_plot:
    import matplotlib.pyplot as plt

    df = pd.read_csv(output_csv_path)
    print(df)
    df = df.set_index("percent").sort_index()
    df.groupby(["stype", "mtype"]).plot()
    #df.plot.scatter(x="percent", y="RMSE")

    plt.show()
