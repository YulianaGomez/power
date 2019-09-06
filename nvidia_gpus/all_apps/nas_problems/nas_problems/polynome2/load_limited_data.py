import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

def load_data():
	path = '/home/yzamora/power_update/nvidia_gpus/all_apps/'
	p100_20per = pd.read_csv(path + 'AM_AL_spec_P100_20Per.csv')
	v100_20per = pd.read_csv(path + 'AM_AL_spec_V100_20Per.csv')
	p100_20per_d = p100_20per.drop(columns=['architecture','input','application_name','kernelname'])

	p100_20per_vals = p100_20per_d.values
	#p100_20per_scaler = StandardScaler().fit(p100_20per_vals)
	#p100_20per_norm = p100_20per_scaler.transform(p100_20per_vals)

	v100_20per = v100_20per.drop(columns=['architecture','input','application_name','kernelname'])
	v100_20per_vals = v100_20per.values
	#v100_20per_scaler = StandardScaler().fit(v100_20per_vals)
	#v100_20per_norm = v100_20per_scaler.transform(v100_20per_vals)

	X_trainP, X_testP, y_trainV, y_testV = train_test_split(p100_20per_vals, v100_20per_vals, test_size=.33, random_state=42)

	print(X_trainP.shape, y_trainV.shape, X_testP.shape, y_testV.shape)
	return (X_trainP, y_trainV), (X_testP, y_testV)

if __name__ == '__main__':
	load_data()
