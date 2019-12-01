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
# load pima indians dataset
#dataset = pd.read_csv("p100_only.csv", delimiter=",").values
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
# create model
#from sklearn.preprocessing import MinMaxScaler



root_path = '/Users/yzamora/Desktop/ActiveLearningFrameworkTutorial/'
df_P100 = pd.read_csv('../p100_all_data.csv', index_col = 0)
df_p100_ipc = df_P100.drop(columns=['shared_utilization','stall_other','single_precision_fu_utilization','architecture','input','application_name','kernelname'])
df_p100_ipc = df_p100_ipc['ipc']
p100_ipc_values = df_p100_ipc.values
#p100_ipc_values = MinMaxScaler().fit_transform(p100_ipc_values)
df_p100_othermetrics = df_P100.drop(columns=['ipc','architecture','input','application_name','kernelname'])
# When trying to reduce metrics - use below
##df_p100_othermetrics = df_P100[['shared_utilization','stall_other','single_precision_fu_utilization']]

x_scaler = StandardScaler()
X_P = x_scaler.fit_transform(df_p100_othermetrics.values)

y_scaler = StandardScaler()
Y_P = y_scaler.fit_transform(p100_ipc_values[:,None])[:,0]

#Splitting just P100 data
train_test_size = 0.98
XP_train, XP_test, yP_train, yP_test = train_test_split(X_P, Y_P, test_size=train_test_size, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(XP_train, yP_train,test_size=0.3, random_state=42)
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
    model.add(Dense(120, input_dim=115, kernel_initializer='normal',activation='relu'))
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


p_model = my_model()

p_model.fit(X_train, y_train, epochs=100, batch_size=10000, verbose=1,validation_data=(X_val,y_val))

## Testing results with validation data
##XP_test = X_val
##yP_test = y_val
y_pred = p_model.predict(XP_test)

y_pred = y_scaler.inverse_transform(y_pred)
yP_test = y_scaler.inverse_transform(yP_test)


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


MAPE_score = mean_absolute_percentage_error(yP_test,y_pred)

# Plotting predicted vs true
print ("R:",  sp.stats.pearsonr(yP_test.flatten(), y_pred.flatten())[0])
print ("MAE:", np.abs(y_pred.flatten() - yP_test.flatten()).mean(), 's')
print ("RMSE:", np.sqrt(np.power(yP_test.flatten()- y_pred.flatten(), 2).mean()), 's')
print("MAPE", MAPE_score)
print("X_train shape ", X_train.shape)
print("Test data shape", yP_test.shape)
fig, ax = plt.subplots()

fig.set_size_inches(15, 15)
import matplotlib

SMALL_SIZE = 15
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

# Make the plot
ax.scatter(yP_test, y_pred, alpha=0.5)

# Make it pretty
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_xlim())

ax.set_xlabel('True IPC', fontsize=14)
ax.set_ylabel('Predicted IPC', fontsize=14)


train_set_per = round((1 - train_test_size) * .7 * 100, 2)
points_used = X_train.shape[0]
#import pdb; pdb.set_trace()
st_points = "{:,d}".format(points_used)
plt.title("Intra-Architecture IPC prediction (P100) - MAPE: " + str(round(MAPE_score,2)) + ", Training Points Used: " + st_points)
# Add in the goal line
ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--');
#plt.savefig("metric_predict.png")
plt.show()
