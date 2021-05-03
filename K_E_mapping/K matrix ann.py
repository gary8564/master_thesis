# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:16:37 2020

@author: CSHuangLab
"""

#%% import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, PReLU, Dropout, Masking, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from glob import glob
import os 
session_conf = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
#Force Tensorflow to use a single thread
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
import random
SEED=520
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
#%% define hyperparameter tuning function
## hyperparameter tuning---random search
#determine layers and nodes
def create_model(layers):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_dim = n_input, kernel_initializer='he_uniform'))
            #model.add(BatchNormalization())
            model.add(PReLU())
            #model.add(Dropout(0.15)
        else:
            model.add(Dense(nodes, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer='he_uniform'))
            #model.add(BatchNormalization())
            model.add(PReLU())
            #model.add(Dropout(0.15)
    model.add(Dense(8))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def multilayers_model(layers,nodes):
    model = Sequential()
    if layers==1:
        model.add(Dense(nodes, input_dim = n_input, kernel_initializer='he_uniform'))
        #model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dropout(0.15)
    elif layers==2:
        model.add(Dense(nodes, input_dim = n_input, kernel_initializer='he_uniform'))
        #model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dropout(0.15)
        model.add(Dense(nodes, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer='he_uniform'))
        #model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dropout(0.15)
    else:
        model.add(Dense(nodes, input_dim = n_input, kernel_initializer='he_uniform'))
        #model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dropout(0.15)
        model.add(Dense(nodes, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer='he_uniform'))
        #model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dropout(0.15)
        model.add(Dense(nodes, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer='he_uniform'))
        #model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dropout(0.15)
    model.add(Dense(8))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# determine learning rate
def create_onelayer_model(nodes):
    model = Sequential()
    model.add(Dense(nodes, input_dim = n_input, kernel_initializer='he_uniform'))
    #model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.15)
    model.add(Dense(8))
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    return model
#%% Data preprocessing
# E=204000MPa
E=2.04
e1=2.034728
e2=2.03727
e3=2.043232
e4=2.037845
e5=2.011938
e6=1.999624
e7=2.068404
e8=2.093262
path='./damage combination'
'''
### read all files
filenames=sorted(glob('./damage combination/K1*.dat'))
dataframes=[pd.read_table(f) for f in filenames]
'''
### train
#intact
filename=os.path.join(path,'K1.dat')
K_health = np.loadtxt(filename)
K_health = K_health/1000
health_diag = np.diag(K_health)
health_subdiag1 = np.diag(K_health, k=1)
health_subdiag2 = np.diag(K_health, k=2)
health_x = np.hstack((health_diag, health_subdiag1, health_subdiag2))
health_x=health_x.reshape((1,-1))
#health_y=np.full((1,8), E)
health_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))

# 1/2/3/4f reduced by 5%/10%/15%/20%/25%/30%
X_1damage=list()
y_1damage=list()
for i in range(4):
    for j in range(5,31,5):
        filename = os.path.join(path,'K1_'+str(i+1)+'F_'+str(j)+'%.dat')
        K=np.loadtxt(filename)
        K=K/1000
        reduced_diag = np.diag(K)
        reduced_subdiag1 = np.diag(K, k=1)
        reduced_subdiag2 = np.diag(K, k=2)
        reduced_x = np.hstack((reduced_diag, reduced_subdiag1, reduced_subdiag2))
        reduced_x=reduced_x.reshape((1,-1))
        reduced_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))
        reduced_y[:,7-i]=reduced_y[:,7-i]*(100-j)/100
        X_1damage.append(reduced_x)
        y_1damage.append(reduced_y)
X_1damage=np.array(X_1damage)
y_1damage=np.array(y_1damage)
# 2 floor damage
X_2damage, y_2damage=list(), list()
for i in range(4):
    for j in range(i+1,i+4):
        for k in range(5,31,5):
            for l in range(5,31,5):
                try:
                    filename = os.path.join(path,'K1_'+str(i+1)+'F_'+str(k)+'%_'+str(j+1)+'F_'+str(l)+'%.dat')
                    K=np.loadtxt(filename)
                    K=K/1000
                    reduced_diag = np.diag(K)
                    reduced_subdiag1 = np.diag(K, k=1)
                    reduced_subdiag2 = np.diag(K, k=2)
                    reduced_x = np.hstack((reduced_diag, reduced_subdiag1, reduced_subdiag2))
                    reduced_x=reduced_x.reshape((1,-1))
                    reduced_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))
                    reduced_y[:,7-i]=reduced_y[:,7-i]*(100-k)/100
                    reduced_y[:,7-j]=reduced_y[:,7-j]*(100-l)/100
                    X_2damage.append(reduced_x)
                    y_2damage.append(reduced_y)
                except:
                    continue
X_2damage=np.array(X_2damage)
y_2damage=np.array(y_2damage)
# 3 floor damage
X_3damage, y_3damage=list(), list()
for i in range(5,31,5):
    filename = os.path.join(path,'K1_1.2.3F_'+str(i)+'%.dat')
    K=np.loadtxt(filename)
    K=K/1000
    reduced_diag = np.diag(K)
    reduced_subdiag1 = np.diag(K, k=1)
    reduced_subdiag2 = np.diag(K, k=2)
    reduced_x = np.hstack((reduced_diag, reduced_subdiag1, reduced_subdiag2))
    reduced_x=reduced_x.reshape((1,-1))
    reduced_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))
    reduced_y[:,5:8]=reduced_y[:,5:8]*(100-i)/100
    X_3damage.append(reduced_x)
    y_3damage.append(reduced_y)

for i in range(5,31,5):
    filename = os.path.join(path,'K1_1.2.4F_'+str(i)+'%.dat')
    K=np.loadtxt(filename)
    K=K/1000
    reduced_diag = np.diag(K)
    reduced_subdiag1 = np.diag(K, k=1)
    reduced_subdiag2 = np.diag(K, k=2)
    reduced_x = np.hstack((reduced_diag, reduced_subdiag1, reduced_subdiag2))
    reduced_x=reduced_x.reshape((1,-1))
    reduced_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))
    reduced_y[:,4]=reduced_y[:,4]*(100-i)/100
    reduced_y[:,6:8]=reduced_y[:,6:8]*(100-i)/100
    X_3damage.append(reduced_x)
    y_3damage.append(reduced_y)

for i in range(5,31,5):
    filename = os.path.join(path,'K1_1.3.4F_'+str(i)+'%.dat')
    K=np.loadtxt(filename)
    K=K/1000
    reduced_diag = np.diag(K)
    reduced_subdiag1 = np.diag(K, k=1)
    reduced_subdiag2 = np.diag(K, k=2)
    reduced_x = np.hstack((reduced_diag, reduced_subdiag1, reduced_subdiag2))
    reduced_x=reduced_x.reshape((1,-1))
    reduced_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))
    reduced_y[:,4:6]=reduced_y[:,4:6]*(100-i)/100
    reduced_y[:,7]=reduced_y[:,7]*(100-i)/100
    X_3damage.append(reduced_x)
    y_3damage.append(reduced_y)

for i in range(5,31,5):
    filename = os.path.join(path,'K1_2.3.4F_'+str(i)+'%.dat')
    K=np.loadtxt(filename)
    K=K/1000
    reduced_diag = np.diag(K)
    reduced_subdiag1 = np.diag(K, k=1)
    reduced_subdiag2 = np.diag(K, k=2)
    reduced_x = np.hstack((reduced_diag, reduced_subdiag1, reduced_subdiag2))
    reduced_x=reduced_x.reshape((1,-1))
    reduced_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))
    reduced_y[:,4:7]=reduced_y[:,4:7]*(100-i)/100
    X_3damage.append(reduced_x)
    y_3damage.append(reduced_y)
X_3damage=np.array(X_3damage)
y_3damage=np.array(y_3damage)

# 4 flor damage
X_4damage, y_4damage=list(), list()
for i in range(5,31,5):
    filename = os.path.join(path,'K1_1.2.3.4F_'+str(i)+'%.dat')
    K=np.loadtxt(filename)
    K=K/1000
    reduced_diag = np.diag(K)
    reduced_subdiag1 = np.diag(K, k=1)
    reduced_subdiag2 = np.diag(K, k=2)
    reduced_x = np.hstack((reduced_diag, reduced_subdiag1, reduced_subdiag2))
    reduced_x=reduced_x.reshape((1,-1))
    reduced_y=np.array([e8, e7, e6, e5, e4, e3, e2, e1]).reshape((1,8))
    reduced_y[:,4:8]=reduced_y[:,4:8]*(100-i)/100
    X_4damage.append(reduced_x)
    y_4damage.append(reduced_y)
X_4damage=np.array(X_4damage)
y_4damage=np.array(y_4damage)
# concatenate
X_1damage=np.squeeze(X_1damage)
X_2damage=np.squeeze(X_2damage)
X_3damage=np.squeeze(X_3damage)
X_4damage=np.squeeze(X_4damage)
y_1damage=np.squeeze(y_1damage)
y_2damage=np.squeeze(y_2damage)
y_3damage=np.squeeze(y_3damage)
y_4damage=np.squeeze(y_4damage)
X_train = np.vstack((health_x,X_1damage,X_2damage,X_3damage,X_4damage))
y_train = np.vstack((health_y,y_1damage,y_2damage,y_3damage,y_4damage))
n_input = X_train.shape[1]
print('trainX:',X_train.shape)
print('trainy:',y_train.shape)


### test
filename=os.path.join(path,'simulation_K_1F_3F_80%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x1 = np.hstack((diag, subdiag1, subdiag2))
x1 = x1.reshape((1,-1))
y1 = np.full((1,8), E)
y1[:,7]=y1[:,7]*0.8
y1[:,5]=y1[:,5]*0.8

filename=os.path.join(path,'simulation_K_1F_3F_90%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x2 = np.hstack((diag, subdiag1, subdiag2))
x2 = x2.reshape((1,-1))
y2 = np.full((1,8), E)
y2[:,7]=y2[:,7]*0.9
y2[:,5]=y2[:,5]*0.9

filename=os.path.join(path,'simulation_K_1F_80%E_3F_90%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x3 = np.hstack((diag, subdiag1, subdiag2))
x3 = x3.reshape((1,-1))
y3 = np.full((1,8), E)
y3[:,7]=y3[:,7]*0.8
y3[:,5]=y3[:,5]*0.9

filename=os.path.join(path,'simulation_K_1F_90%E_3F_80%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x4 = np.hstack((diag, subdiag1, subdiag2))
x4 = x4.reshape((1,-1))
y4 = np.full((1,8), E)
y4[:,7]=y4[:,7]*0.9
y4[:,5]=y4[:,5]*0.8

filename=os.path.join(path,'simulation_K_1F_90%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x5 = np.hstack((diag, subdiag1, subdiag2))
x5 = x5.reshape((1,-1))
y5 = np.full((1,8), E)
y5[:,7]=y5[:,7]*0.9

filename=os.path.join(path,'simulation_K_1F_80%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x6 = np.hstack((diag, subdiag1, subdiag2))
x6 = x6.reshape((1,-1))
y6 = np.full((1,8), E)
y6[:,7]=y6[:,7]*0.8

filename=os.path.join(path,'simulation_K_3F_80%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x7 = np.hstack((diag, subdiag1, subdiag2))
x7 = x7.reshape((1,-1))
y7 = np.full((1,8), E)
y7[:,5]=y7[:,5]*0.8

filename=os.path.join(path,'simulation_K_3F_90%E.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x8 = np.hstack((diag, subdiag1, subdiag2))
x8 = x8.reshape((1,-1))
y8 = np.full((1,8), E)
y8[:,5]=y8[:,5]*0.9

filename=os.path.join(path,'K1_2F_10%_3F_30%_4F_20%.dat')
K = np.loadtxt(filename)
K = K/1000
diag = np.diag(K)
subdiag1 = np.diag(K, k=1)
subdiag2 = np.diag(K, k=2)
x9 = np.hstack((diag, subdiag1, subdiag2))
x9 = x9.reshape((1,-1))
y9 = health_y
y9[:,4]=y9[:,4]*0.8
y9[:,5]=y9[:,5]*0.7
y9[:,6]=y9[:,6]*0.9



X_test = np.vstack((x1,x2,x3,x4,x5,x6,x7,x8,x9))
y_test = np.vstack((y1,y2,y3,y4,y5,y6,y7,y8,y9))
print('testX:',X_test.shape)
print('testy:',y_test.shape)
#%%
epochs=500
batch_size=8
model = Sequential()
model.add(Dense(16, activation='softplus', input_dim = n_input, kernel_initializer='he_uniform'))
#model.add(PReLU())
#model.add(Dropout(0.15))
#model.add(Dense(16, activation='hard_sigmoid'))#, kernel_initializer='he_uniform'))
#model.add(PReLU())
#model.add(Dropout(0.15))
model.add(Dense(8))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=1)

'''
# Hyperparameter tuning
# random search
model = KerasRegressor(build_fn=create_onelayer_model, verbose=1, epochs=500)
# define the grid search parameters
#layers=[[16], [16, 8], [16, 8, 4]]
batch_size = [1,2,4,8,16]
params = dict(nodes=[16,32,64,128], batch_size=batch_size)
random = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='neg_mean_absolute_error',n_jobs=1)
#lrate = LearningRateScheduler(step_decay)
#loss_history = LossHistory()
#callbacks_list = [lrate, loss_history]
random_result = random.fit(X_train, y_train, validation_data=(X_val, y_val))
# summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
means = random_result.cv_results_['mean_test_score']
stds = random_result.cv_results_['std_test_score']
params = random_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

model = KerasRegressor(build_fn=multilayers_model, verbose=1, epochs=500)
# define the grid search parameters
#layers=[[16], [16, 8], [16, 8, 4]]
batch_size = [1,2,4,8,16,32]
params = dict(layers=[1,2,3], nodes=[16,32,64,128], batch_size=batch_size)
random = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='neg_mean_absolute_error',n_jobs=1)
#lrate = LearningRateScheduler(step_decay)
#loss_history = LossHistory()
#callbacks_list = [lrate, loss_history]
random_result = random.fit(X_train, y_train, validation_data=(X_val, y_val))
# summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
means = random_result.cv_results_['mean_test_score']
stds = random_result.cv_results_['std_test_score']
params = random_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

#%% training visualization
# plot history
'''
# for random search
model = random_result.best_estimator_.model 
model.summary()
plt.plot(random_result.best_estimator_.model.history.history['loss'], label='train')
plt.plot(random_result.best_estimator_.model.history.history['val_loss'], label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Training Loss')
plt.legend()
plt.show()
'''
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Training Loss')
plt.legend()
plt.show()

train_predict = model.predict(X_train, verbose=0)
print(train_predict)
train_error=mean_absolute_error(y_train, train_predict)
print('training mean absolute error:',train_error)

yhat = model.predict(X_test, verbose=0)
print(yhat)
test_error=mean_absolute_error(y_test, yhat)
print('test mean absolute error:',test_error)