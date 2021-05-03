# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:41:55 2020

@author: CSHuangLab
"""


# %% import modules
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import os
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, PReLU, Dropout, Masking, Flatten
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import scipy
import seaborn as sns
from tensorflow.keras.utils import plot_model
from scipy import stats
session_conf = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
#Force Tensorflow to use a single thread
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
import random
SEED=4134
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% define function        
class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss') <= 5e-07):
            print("\n\n\nReached loss value so cancelling training!\n\n\n")
            self.model.stop_training = True
        
# learning rate decay function
'''
def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.05
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate
'''  
def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 20.0
   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate

# custom callbacks--loss history
class LossHistory(K.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))

# Average time series to decrease noises
def decrease_noises(sequence, time_steps):
    seq=list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + time_steps
		# check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        mean=np.mean(sequence[i:end_ix])
        seq.append(mean)
    return np.array(seq)

# Split a multivariate sequence into samples
def split_multisequence(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X = list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x = sequence[i:end_ix]
		X.append(seq_x)
	return np.array(X)
###LSTM model function
## training model function
## hyperparameter tuning---random search
#determine layers and nodes
def create_model(layers,n_input):
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
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# determine learning rate
def create_onelayer_model(nodes):
    model = Sequential()
    model.add(Dense(nodes, input_dim = n_input, kernel_initializer='he_uniform'))
    #model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.15)
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
# MLP
def mlp_model(epochs, n_input, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim = n_input, kernel_initializer='he_uniform'))
    #model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(0.15))
    #model.add(Dense(64, kernel_initializer='he_uniform'))
    #model.add(BatchNormalization())
    #model.add(PReLU())
    #model.add(Dropout(0.15))
    #model.add(Dropout(0.15))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
#%% load training data
# train
while True:
    try:
        trainX_file = input("Please type the desired file name of trainX:")
        trainX=np.load(trainX_file)
        break
    except IOError:
        print('Error...please check your typing or current directory...file name should include .npy')
        continue
while True:
    try:
        trainy_file = input("Please type the desired file name of trainy:")
        trainy=np.load(trainy_file)
        break
    except IOError:
       print('Error...please check your typing or directory...file name should include .npy')
       continue
n_input=trainX.shape[1]
n_features = 8 # number of d.o.f.
n_steps=int((n_input+n_features)/(n_features+1))
'''
# convlstm or cnn-lstm
n_features = trainX.shape[4]
n_seq = np.int(np.sqrt(n_steps))
n_steps = np.int(np.sqrt(n_steps))
'''
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)

#%% Build the model for healthy under different input
#####train##############
###MLP

# epoch and batch size setting
epochs = 500
batch_size = 32
#nodes=int(np.ceil(2/3*(n_input+trainy.shape[1])))
# calling callback functions
## early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
trainingStopCallback = haltCallback()
lrate = LearningRateScheduler(step_decay)
loss_history = LossHistory()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
## checkpoint 
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

## Tensorboard
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)

# create model
# MLP
model = mlp_model(epochs, n_input, trainy.shape[1])

# fit model
history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, 
                    validation_split=0.3, verbose=1)
                   # validation_data=(valX,valy), callbacks=callbacks_list, shuffle=False)

'''
# Hyperparameter tuning
# random search
model = KerasRegressor(build_fn=create_onelayer_model, verbose=1, epochs=500)
# define the grid search parameters
#layers=[[16], [16, 8], [16, 8, 4]]
batch_size = [16,32,64,128]
params = dict(nodes=[16,32,64,128], batch_size=batch_size)
random = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='neg_mean_absolute_error',n_jobs=1)
#lrate = LearningRateScheduler(step_decay)
#loss_history = LossHistory()
#callbacks_list = [lrate, loss_history]
random_result = random.fit(trainX, trainy, validation_split=0.3)
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
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Training Loss')
plt.legend()
plt.show()


#%% training visualization---plot error
train_predict = model.predict(trainX, verbose=0)
print(train_predict)
print(train_predict.shape)
print('training MSE:',mean_squared_error(trainy, train_predict))
print('training R-square:',r2_score(trainy, train_predict))
for i in range(train_predict.shape[1]):
    pred_fl = train_predict[:,i]
    target_train=trainy[:,i]
    t1 = np.linspace(0, len(target_train), len(target_train), endpoint=False)
    t2 = np.linspace(n_steps, len(target_train), len(pred_fl), endpoint=False)
    fig = plt.figure(figsize=(90,36))
    ax1 = plt.subplot(211)
    ax1.plot(t1, target_train, label='observed', linestyle='-', color = 'k', linewidth=7.5)
    ax1.plot(t2, pred_fl, label='forecast', linestyle=':', color = 'b', linewidth=7.5)
    ax1.set_xlim(0,len(target_train))
    ax1.set_ylabel("Absolute acceleration (m/$s^2$)",fontsize=80)
    ax1.set_title("%d floor" %(i+1),fontsize=125)
    ax1.legend(loc=1,prop={"size":80})
    ax1.spines['left'].set_linewidth(5)
    ax1.spines['right'].set_linewidth(5)
    ax1.spines['bottom'].set_linewidth(5)
    ax1.spines['top'].set_linewidth(5)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    ax2 = plt.subplot(212, sharex=ax1)
    # plot the error on the training set
    train_abs_error = np.zeros(pred_fl.shape)
    trainy_fl = trainy[:,i]
    for j in range(len(pred_fl)):
        train_abs_error[j] = np.abs((pred_fl[j] - trainy_fl[j]))
    ax2.plot(train_abs_error, linewidth=7.5)
    ax2.set_ylim(0,train_abs_error.max())
    ax2.set_xlabel("Number of samples",fontsize=80)
    ax2.set_ylabel("Absolute error",fontsize=80)
    ax2.spines['left'].set_linewidth(5)
    ax2.spines['right'].set_linewidth(5)
    ax2.spines['bottom'].set_linewidth(5)
    ax2.spines['top'].set_linewidth(5)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.show()
    print('training MSE for %d floor:'%(i+1),mean_squared_error(trainy_fl, pred_fl))
    #fig.savefig('training(mlp)_%d_floor.png'%(i+1), transparent=False)

#%% TEST (healthy state)
# summarize model.
model.summary()
path='./no noise'
test_kobe = np.loadtxt(os.path.join(path,'00%_kobe_responses_1_8DOF_500Hz_a.dat'))
test_kobe = test_kobe[2000:6000,:] 
plt.plot(test_kobe)     # from the plot, we can guess the acceleration units should be g
plt.show()

test_northridge = np.loadtxt(os.path.join(path, '00%_northridge_responses_1_8DOF_500Hz_a.dat'))
test_northridge=test_northridge[250:4000]
plt.plot(test_northridge)     
plt.show()

test_turkey = np.loadtxt(os.path.join(path, '00%_turkey_responses_1_8DOF_500Hz_a.dat'))
test_turkey=test_turkey[3500:6000,:]
plt.plot(test_turkey)    
plt.show()

test_italy = np.loadtxt(os.path.join(path, '00%_09italy_responses_1_8DOF_500Hz_a.dat'))
test_italy=test_italy[3500:4750]
plt.plot(test_italy)
plt.show()

test_hualien = np.loadtxt(os.path.join(path, '00%_180206hualien_responses_1_8DOF_500Hz_a.dat'))
test_hualien=test_hualien[2700:4000]
plt.plot(test_hualien)
plt.show()
'''
# 10% noises
path='./10% noises'
test_kobe = np.loadtxt(os.path.join(path, '10%_kobe_responses_1_8DOF_500Hz_a.dat'))
test_kobe = test_kobe[2000:6000,:] 

test_northridge = np.loadtxt(os.path.join(path, '10%_northridge_responses_1_8DOF_500Hz_a.dat'))
test_northridge=test_northridge[250:4000,:]

test_turkey = np.loadtxt(os.path.join(path, '10%_turkey_responses_1_8DOF_500Hz_a.dat'))
test_turkey=test_turkey[3500:6000,:]

test_italy = np.loadtxt(os.path.join(path,'10%_italy09_responses_1_8DOF_500Hz_a.dat'))
test_italy=test_italy[3000:6000,:]

test_hualien = np.loadtxt(os.path.join(path,'10%_hualien2018_responses_1_8DOF_500Hz_a.dat'))
test_hualien=test_hualien[2500:4000,:]

# 5% noises
path='./5% noises'
test_kobe = np.loadtxt(os.path.join(path,'05%_kobe_responses_1_8DOF_500Hz_a.dat'))
test_kobe = test_kobe[2000:6000,:] 

test_northridge = np.loadtxt(os.path.join(path,'05%_northridge_responses_1_8DOF_500Hz_a.dat'))
test_northridge=test_northridge[250:4000,:]

test_turkey = np.loadtxt(os.path.join(path,'05%_turkey_responses_1_8DOF_500Hz_a.dat'))
test_turkey=test_turkey[3500:6000,:]

test_italy = np.loadtxt(os.path.join(path,'05%_09italy_responses_1_8DOF_500Hz_a.dat'))
test_italy=test_italy[3000:6000,:]

test_hualien = np.loadtxt(os.path.join(path,'05%_hualien2018_responses_1_8DOF_500Hz_a.dat'))
test_hualien=test_hualien[2500:5000,:]
'''
# test data
test1=test_kobe
#test1=test_northridge
#test1 = test_turkey
#test1 = test_italy
#test1 = test_hualien
plt.plot(test1)    
plt.show()
print(test1.shape)
#%%relative acc
for i in range(test1.shape[1]-1):
    test1[:,i]=test1[:,i]-test1[:,8]
#%% Normalization
test_data = test1.T
'''
# denoise
for i in range(test_data.shape[0]):
    test_temp=decrease_noises(test_data[i], time_steps=5)
    test_data[i,:len(test_temp)]=test_temp
'''
# normalization(scaling data in a range of -1~1)
test_matrix=test_data.T/np.max(abs(test_data))
test_input = test_matrix[:,8]
first_test = test_matrix[:,7]
second_test = test_matrix[:,6]
third_test = test_matrix[:,5]
fourth_test = test_matrix[:,4]
fifth_test = test_matrix[:,3]
sixth_test = test_matrix[:,2]
seventh_test = test_matrix[:,1]
eighth_test = test_matrix[:,0]

#%% test data split into time steps
# convert to [rows, columns] structure
test_input = test_input.flatten()
first_test = first_test.flatten()
second_test = second_test.flatten()
third_test = third_test.flatten()
fourth_test = fourth_test.flatten()
fifth_test = fifth_test.flatten()
sixth_test = sixth_test.flatten()
seventh_test = seventh_test.flatten()
eighth_test = eighth_test.flatten()
test_input = test_input.reshape((len(test_input), 1))
first_test = first_test.reshape((len(first_test),1))
second_test = second_test.reshape((len(second_test),1))
third_test = third_test.reshape((len(third_test),1))
fourth_test = fourth_test.reshape(len(fourth_test),1)
fifth_test = fifth_test.reshape((len(fifth_test),1))
sixth_test = sixth_test.reshape((len(sixth_test),1))
seventh_test = seventh_test.reshape((len(seventh_test),1))
eighth_test = eighth_test.reshape((len(eighth_test), 1))
# horizontally stack columns
test = np.hstack((first_test, second_test, third_test, fourth_test, fifth_test, sixth_test, seventh_test, eighth_test))
#test = np.hstack((first_test, third_test, fifth_test, eighth_test))
# split sequences by time steps
testX1, testy = split_multisequence(test, n_steps-1)
testX2 = split_sequence(test_input, n_steps)
print(testX1.shape, testX2.shape, testy.shape)
# reshape from [samples, timesteps] into [samples, timesteps, features]
testX2 = np.squeeze(testX2)
# flatten input
n_input_temp = testX1.shape[1] * testX1.shape[2]
testX1 = testX1.reshape((testX1.shape[0], n_input_temp))
# concatenate along columns to assemble trainX, testX
testX_health = np.concatenate((testX1, testX2),axis=1)
testy_health=testy
print('testX:',testX_health.shape)
print('testy:',testy.shape)

#%% Plot test results(HEALTH)
# predict test data
# test
yhat = model.predict(testX_health, verbose=0)
print(yhat)
print(yhat.shape)
print('test MSE:',mean_squared_error(testy, yhat))
print('test R-square:',r2_score(testy, yhat))
h=1/500
# demonstrate test prediction
for i in range(yhat.shape[1]):
    # test
    pred_fl = yhat[:-15,i]
    target = testy_health[:-15,i]
    t1 = h*np.linspace(0, len(target), len(target), endpoint=False)
    t2 = h*np.linspace(0, len(target), len(pred_fl), endpoint=False)
    fig = plt.figure(figsize=(90,36))
    ax1 = plt.subplot(211)
    ax1.plot(t1, target, label='observed', linestyle='-', color = 'k', linewidth=7.5)
    ax1.plot(t2, pred_fl, label='forecast', linestyle=':', color = 'b', linewidth=7.5)
    ax1.set_xlim(0,h*len(target))
    ax1.set_ylabel("Absolute acceleration (m/$s^2$)",fontsize=80)
    ax1.set_title("%d floor" %(i+1),fontsize=125)
    ax1.legend(loc=1,prop={"size":80})
    ax1.spines['left'].set_linewidth(5)
    ax1.spines['right'].set_linewidth(5)
    ax1.spines['bottom'].set_linewidth(5)
    ax1.spines['top'].set_linewidth(5)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    ax2 = plt.subplot(212, sharex=ax1)
    # plot the error on the test set
    abs_error = np.zeros(pred_fl.shape)
    testy_fl = testy[:-15,i]
    for j in range(len(pred_fl)):
        abs_error[j] = np.abs((pred_fl[j] - testy_fl[j]))
    ax2.plot(t2, abs_error, linewidth=7.5)
    ax2.set_xlim(0,h*len(target))
    ax2.set_ylim(0,abs_error.max())
    ax2.set_xlabel("Time(sec)", fontsize=80)
    ax2.set_ylabel("Absolute error", fontsize=80)
    ax2.spines['left'].set_linewidth(5)
    ax2.spines['right'].set_linewidth(5)
    ax2.spines['bottom'].set_linewidth(5)
    ax2.spines['top'].set_linewidth(5)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.show()
    print('test MSE for %d floor:'%(i+1),mean_squared_error(testy_fl, pred_fl))


#%% Test (damaged)
# summarize model.
model.summary()
# test data
path='./no noise'
test1 = np.loadtxt(os.path.join(path,'00%_kobe_responses_1_E70%_1_8DOF_500Hz_a.dat'))
#test1 = np.loadtxt(os.path.join(path,'10%_kobe_responses_1_3_E70%_1_8DOF_500Hz_a.dat')) #10%noise
#test1 = np.loadtxt(os.path.join(path,'05%_kobe_responses_1.3_E70%_1_8DOF_500Hz_a.dat')) #5%noises
plt.plot(test1)
plt.show()
print(test1.shape)

#%%relative acc
for i in range(test1.shape[1]-1):
    test1[:,i]=test1[:,i]-test1[:,8]
#%% normalization
# cut the noise before earthquake shock
test1=test1[2000:6000,:] # kobe
#test1=test1[250:4000,:] # northridge
#test1=test1[3500:6000,:] # turkey
#test1=test1[3500:4750,:] #italy
#test1=test1[2700:4000] #hualien
# select input, 1st, and 8th floor to predict 8th floor
# data preprocessing (units:g)
# scaler = MinMaxScaler(feature_range=(-1,1))
test_data = test1.T
'''
# denoise
for i in range(test_data.shape[0]):
    test_temp=decrease_noises(test_data[i], time_steps=5)
    test_data[i,:len(test_temp)]=test_temp
'''
# normalization(scaling data in a range of -1~1)
test_matrix=test_data.T/np.max(abs(test_data))
test_input = test_matrix[:,8]
first_test = test_matrix[:,7]
second_test = test_matrix[:,6]
third_test = test_matrix[:,5]
fourth_test = test_matrix[:,4]
fifth_test = test_matrix[:,3]
sixth_test = test_matrix[:,2]
seventh_test = test_matrix[:,1]
eighth_test = test_matrix[:,0]

# convert to [rows, columns] structure
test_input = test_input.flatten()
first_test = first_test.flatten()
second_test = second_test.flatten()
third_test = third_test.flatten()
fourth_test = fourth_test.flatten()
fifth_test = fifth_test.flatten()
sixth_test = sixth_test.flatten()
seventh_test = seventh_test.flatten()
eighth_test = eighth_test.flatten()
test_input = test_input.reshape((len(test_input), 1))
first_test = first_test.reshape((len(first_test),1))
second_test = second_test.reshape((len(second_test),1))
third_test = third_test.reshape((len(third_test),1))
fourth_test = fourth_test.reshape(len(fourth_test),1)
fifth_test = fifth_test.reshape((len(fifth_test),1))
sixth_test = sixth_test.reshape((len(sixth_test),1))
seventh_test = seventh_test.reshape((len(seventh_test),1))
eighth_test = eighth_test.reshape((len(eighth_test), 1))
# horizontally stack columns
test = np.hstack((first_test, second_test, third_test, fourth_test, fifth_test, sixth_test, seventh_test, eighth_test))
#test = np.hstack((first_test, third_test, fifth_test, eighth_test))
# split sequences by time steps
testX1, testy = split_multisequence(test, n_steps-1)
testX2 = split_sequence(test_input, n_steps)
print(testX1.shape, testX2.shape, testy.shape)
# reshape from [samples, timesteps] into [samples, timesteps, features]
testX2 = np.squeeze(testX2)
# flatten input
testX1 = testX1.reshape((testX1.shape[0], n_input_temp))
# concatenate along columns to assemble trainX, testX
testX = np.concatenate((testX1, testX2),axis=1)
#print('trainX:',trainX.shape)
#print('trainy:',trainy.shape)
print('testX:',testX.shape)
print('testy:',testy.shape)
testX_damage=testX
testy_damage=testy
#%% TEST(damage)
# predict test data
yhat_h = model.predict(testX_health, verbose=0)
yhat_d = model.predict(testX_damage, verbose=0)
#print(yhat)
#print(yhat.shape)
print('test MSE:',mean_squared_error(testy_damage, yhat_d))
print('test R-square:',r2_score(testy_damage, yhat_d))
# demonstrate test prediction
h=1/500
for i in range(yhat_d.shape[1]):
    pred_fl = yhat_d[:-5,i]
    target = testy[:-5,i]
    t1 = h*np.linspace(0, len(target), len(target), endpoint=False)
    t2 = h*np.linspace(0, len(target), len(pred_fl), endpoint=False)
    fig = plt.figure(figsize=(90,36))
    ax1 = plt.subplot(211)
    ax1.plot(t1, target, label='observed', linestyle='-', color = 'k', linewidth=7.5)
    ax1.plot(t2, pred_fl, label='forecast', linestyle=':', color = 'b', linewidth=7.5)
    ax1.set_xlim(0,len(target)*h)
    ax1.set_ylabel("Absolute acceleration (m/$s^2$)",fontsize=80)
    ax1.set_title("%d floor (1st & 3rd floor damage)" %(i+1),fontsize=125)
    ax1.legend(loc=1,prop={"size":80})
    ax1.spines['left'].set_linewidth(5)
    ax1.spines['right'].set_linewidth(5)
    ax1.spines['bottom'].set_linewidth(5)
    ax1.spines['top'].set_linewidth(5)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    ax2 = plt.subplot(212, sharex=ax1)
    # plot the error on the test set
    abs_error = np.zeros(pred_fl.shape)
    testy_fl = testy[:-5,i]
    for j in range(len(pred_fl)):
        abs_error[j] = np.abs((pred_fl[j] - testy_fl[j]))
    ax2.plot(t2,abs_error, linewidth=7.5)
    #ax2.set_ylim(0,0.0175)
    ax2.set_ylim(0,abs_error.max())
    ax2.set_xlabel("Time(sec)", fontsize=80)
    ax2.set_ylabel("Absolute error", fontsize=80)
    ax2.spines['left'].set_linewidth(5)
    ax2.spines['right'].set_linewidth(5)
    ax2.spines['bottom'].set_linewidth(5)
    ax2.spines['top'].set_linewidth(5)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.show()
    print('test MSE for %d floor:'%(i+1),mean_squared_error(testy_fl, pred_fl))

#%% save results
model.save('mlp.h5')
train_error=train_predict-trainy
error_health=yhat_h-testy_health
error_damaged=yhat_d-testy_damage
np.save('train_error_mlp',train_error)
np.save('test_error_health_mlp', error_health)
np.save('test_error_damaged_mlp', error_damaged)
