# Load libraries
import numpy as np
from keras.layers import Input,Dense
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import pandas as pd
import sys
import os
import glob
import csv
import os.path

# Step 1 - Process Data:

save_fname = "saved_data"
if os.path.isfile(save_fname+".npy"):
    print("Loading saved X data.")
    X = np.load(save_fname+".npy")

else:
    #calculating number of kernels total
    kernel_count = 0
    total_kernels = []
    combined_data_ = {}
    target_kernels = []
    metric_targets = []
    all_sig_metrics = []
    for filen_ in glob.glob("/Users/yzamora/Desktop/performance_data/p100_testing/five_apps/*.csv"):

        filen = os.path.basename(filen_)
        filen_split = filen.split('.')[0].split('_')
        bench_name = filen_split[0]
        size = filen_split[1]

        key_root = bench_name
        levels = ["Idle", "Low","High", "Max"]
        bw_units = ["GB", "MB", "KB" ,"0B"]

        # Now open the file and look for the data
        with open(filen_ ,'r') as file_handle:

            data_found = False
            ncols = 1
            fdata = csv.reader(file_handle)
            index_lookup = {}
            for line_split in fdata:

                lsplt = (len(line_split) > 0)

                if data_found:

                    if lsplt and len(line_split) == ncols:

                        # Read in desired value for the current metric
                        target_index = index_lookup['Avg']; value = 0
                        metric_name = line_split[index_lookup['Metric Name']]
                        if line_split[target_index].isdecimal():
                            if line_split[target_index]!= '0':

                                all_sig_metrics.append(metric_name)
                                value = int(line_split[ target_index ])

                            # Labeled with percentage
                        elif "%" == line_split[target_index][-1]:
                            #print ("percentage loop")
                            all_sig_metrics.append(metric_name)
                            value = float(line_split[ target_index ][0:7]) / 100.0

                        # Labeled with bandwidth units
                        elif line_split[ target_index ][-4:-2] in bw_units:
                            # Just take the first
                            units = line_split[ target_index ][-4:-2]
                            all_sig_metrics.append(metric_name)
                            mfact = 1.0
                            if   units == "KB": mfact = 1024
                            elif units == "MB": mfact = 1024*1024
                            elif units == "GB": mfact = 1024*1024*1024
                            elif units == "0B":  mfact = 1
                            value = float(line_split[ target_index ][0:7]) * mfact

                        # idle, low, max
                        elif line_split[ target_index ][-1] == ")":
                            #print ("low")
                            all_sig_metrics.append(metric_name)
                            value = int(line_split[ target_index].split('(')[1].split(")")[0])

                        # otherwise, float
                        else:
                            value = float(line_split[ target_index ])

                         # Parse name of kernel
                        kernel_name = line_split[ index_lookup['Kernel'] ].split('(')[0]
                        if not(kernel_name in total_kernels):
                            total_kernels.append(kernel_name)
                            kernel_count += 1

                        # Define kernel-specific key
                        key = key_root + size + "_" + kernel_name

                        # Initialize dict for this key, if it is new
                        if not (key in combined_data_):
                            combined_data_ [ key ] = {}
                        if not (kernel_name in target_kernels):
                            target_kernels.append(kernel_name)

                        # Store value for the metric being read right now
                        combined_data_[key][ metric_name ] = value
                        combined_data_[key]["kernelname"] = kernel_name

                    else: data_found = False


                elif lsplt and line_split[0] == 'Device' and line_split[1] == 'Kernel':
                    # Set flag that we are at the data:
                    data_found = True
                    # Set number of columns in table:
                    ncols = len(line_split)
                    # Generate an index lookup table:
                    idx = 0
                    for term in line_split:
                        index_lookup[term] = idx
                        idx += 1

    #from sklearn.preprocessing import scale
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    df = pd.DataFrame.from_dict(combined_data_,orient='index')
    df = df.dropna(axis=1,how='any')
    df2 = df.drop(columns=['kernelname'])
    data = df2.values #scale(df2.values)
    #X = StandardScaler().fit_transform(data)
    X = MinMaxScaler().fit_transform(data)
    #X=X.astype('float32')/float(X.max())

    # USING FIRST 25 FEATURES FOR NOW...
    X = X[:,:115]
    print(X.shape)
    np.save(save_fname, X)

n_known = 25 # How many features do we know a priori
frac_test = 0.2 # Fraction of data points to use for testing/validation
ntest = int(X.shape[0] * frac_test)
ntrain = int(X.shape[0] - ntest)

# X_train_in will have the first n_known features set X_train
X_train = X[:ntrain]
X_train_in = np.zeros(shape=X_train.shape)
X_train_in[:,:n_known] = X_train[:,:n_known]

# X_test_in will have the first n_known features set to X_test
X_test = X[ntrain:]
X_test_in = np.zeros(shape=X_test.shape)
X_test_in[:,:n_known] = X_test[:,:n_known]

#print(X_test_in[0])
#sys.exit(0)
# Load and scale the data
#(X_train,_), (X_test,_) = mnist.load_data()
#X_train=X_train.astype('float32')/float(X_train.max())
#X_test=X_test.astype('float32')/float(X_test.max())

# Flatten images
#X_train=X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
#X_test=X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
#X_train_in=X_train_in.reshape((len(X_train_in),np.prod(X_train_in.shape[1:])))
#X_test_in=X_test_in.reshape((len(X_test_in),np.prod(X_test_in.shape[1:])))
print("Training set : ",X_train.shape)
print("Testing set : ",X_test.shape)

# Creating an autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 60

# Defining model (network architecture)
autoencoder=Sequential()
autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,),activation='relu'))
autoencoder.add(Dense(input_dim,activation='sigmoid'))

# Defining the "encoder" (not used in this code, yet)
input_img=Input(shape=(input_dim,))
encoder_layer=autoencoder.layers[0]
encoder=Model(input_img,encoder_layer(input_img))

# define the checkpoint
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Load existing weights if available
if os.path.isfile(filepath):
    print("Loading saved weights.")
    autoencoder.load_weights(filepath)

train = True
if train:
    adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(optimizer=adam, loss='mean_absolute_percentage_error')
    autoencoder.fit(X_train_in, X_train, epochs=5000, batch_size=32, shuffle=True, validation_data=(X_test_in, X_test), callbacks=callbacks_list)

pred = autoencoder.predict(X_test_in)
x = range(pred.shape[1])
print(pred[0])
print(X_test[0])
