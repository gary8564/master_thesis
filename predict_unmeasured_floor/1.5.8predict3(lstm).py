# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:21:37 2020

@author: CSHuangLab
"""


# %% import modules
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, PReLU, Dropout, Masking, Flatten
from tensorflow.keras.layers import LayerNormalization, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, ConvLSTM2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import scipy
import seaborn as sns
session_conf = tf.compat.v1.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
#Force Tensorflow to use a single thread
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
import random
SEED=1996
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% define function
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

# reset lstm state cells function
class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()
        
class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 4e-04):
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

# Split a multivariate sequence into samples
def split_multisequence(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

###LSTM model function
## training model function
# Stateleess LSTM
def stateless_model(epochs, output_dim):
    #adam = K.optimizers.Adam(learning_rate=0.003)
    model = Sequential()
    model.add(LSTM(64, activation='tanh',input_shape=(n_steps,n_features),
                   #kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #bias_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #recurrent_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   return_sequences=False, stateful=False))
    #model.add(LayerNormalization())
    #model.add(PReLU())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1))
    #model.add(LSTM(32, activation='tanh',
                   #kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #bias_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #recurrent_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6), 
                   #bias_regularizer=l2(1e-6), 
                   #return_sequences=False, stateful=False))
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1)
    #model.add(Flatten())
    #model.add(Dense(32, kernel_initializer='he_uniform'))
    #model.add(PReLU())
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
# Stateful LSTM
def stateful_model(epochs, batch_size, output_dim):
    adam = K.optimizers.Adam(learning_rate=0.001)
    model = Sequential()
    model.add(LSTM(32, activation='tanh',
                   #kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #bias_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #recurrent_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   return_sequences=False, stateful=True))
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1))
    #model.add(LSTM(64, activation='tanh',
                   #kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #bias_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #recurrent_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
    #               kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6), 
    #               bias_regularizer=l2(1e-6), return_sequences=False, stateful=True))
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1)
    #model.add(Flatten())
    model.add(Dense(output_dim))
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model
# Bidirectional LSTM
def bidirectional_model(epochs,output_dim):
    #adam = K.optimizers.Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='tanh',
                   #kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #bias_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #recurrent_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   return_sequences=False, stateful=False)))
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1))
    #model.add(Bidireactional(LSTM(32, activation='tanh',
                   #kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #bias_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #recurrent_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
    #               kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6), 
    #               bias_regularizer=l2(1e-6), return_sequences=False, stateful=True)))
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1)
    #model.add(Flatten())
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
# CNN LSTM
def cnnlstm_model(epochs,output_dim,time_steps):
    #adam = K.optimizers.Adam(learning_rate=0.001)
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, time_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=1)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, activation='tanh', return_sequences=False))
    #model.add(LayerNormalization())
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model
# Convlstm
def convlstm_model(epochs,output_dim, n_seq, n_steps):
    #adam = K.optimizers.Adam(learning_rate=0.001)
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(2,2), padding='same', return_sequences=False, 
                         input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(output_dim))
    model.compile(loss='mse', optimizer='adam')
    return model
## hyperparameter tuning---Random Search
#determine layers and nodes
def create_model(layers, layernorm, learn_rate):   
    model = Sequential()
    if layernorm:
        if len(layers)==1:
            model.add(LSTM(layers[0], activation='tanh',return_sequences=False))
            model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1)
        elif len(layers)==2: 
            model.trainable = False
            model.add(LSTM(layers[0], activation='tanh',return_sequences=True))
            model.add(LayerNormalization())
            #model.add(PReLU())
            model.add(LSTM(layers[1], activation='tanh',return_sequences=False))
            model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1))
        else:
            model.trainable = False
            model.add(LSTM(layers[0], activation='tanh',return_sequences=True))
            model.add(LayerNormalization())
            #model.add(PReLU())
            model.add(LSTM(layers[1], activation='tanh',return_sequences=True))
            model.add(LayerNormalization())
            #model.add(PReLU())
            model.add(LSTM(layers[2], activation='tanh',return_sequences=False))
            model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1))
        model.add(Dense(1))
        adam = K.optimizers.Adam(learning_rate=learn_rate)
        model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    else:
        if len(layers)==1:
            model.add(LSTM(layers[0], activation='tanh', return_sequences=False))
            #model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1)
        elif len(layers)==2: 
            model.trainable = False
            model.add(LSTM(layers[0], activation='tanh',
                           return_sequences=True))
            #model.add(LayerNormalization())
            #model.add(PReLU())
            model.add(LSTM(layers[1], activation='tanh',
                           return_sequences=False, 
                           kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6), bias_regularizer=l2(1e-6)))
            #model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1))
        else:
            model.trainable = False
            model.add(LSTM(layers[0], activation='tanh',
                           return_sequences=True))
            #model.add(LayerNormalization())
            #model.add(PReLU())
            model.add(LSTM(layers[1], activation='tanh',
                           return_sequences=True, 
                           kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6), bias_regularizer=l2(1e-6)))
            #model.add(LayerNormalization())
            #model.add(PReLU())
            model.add(LSTM(layers[2], activation='tanh',

                           return_sequences=False, 
                           kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6), bias_regularizer=l2(1e-6)))
            #model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1))
        model.add(Dense(1))
        adam = K.optimizers.Adam(learning_rate=learn_rate)
        model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model
def create_vallina_model(nodes, layernorm):   
    model = Sequential()
    if layernorm:
        model.add(LSTM(nodes, activation='tanh',return_sequences=False))
        model.add(LayerNormalization())
        #model.add(BatchNormalization())
        #model.add(PReLU())
        #model.add(Dropout(0.1)
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(LSTM(nodes, activation='tanh', return_sequences=False))
        #model.add(LayerNormalization())
        #model.add(BatchNormalization())
        #model.add(PReLU())
        #model.add(Dropout(0.1)
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
# determine learning rate
def create_vallina_lstm(lr):   
    model = Sequential()
    model.add(LSTM(32, activation='tanh',return_sequences=False))
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    #model.add(PReLU())
    #model.add(Dropout(0.1)
    model.add(Dense(1))
    adam = K.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
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
n_steps = trainX.shape[1]
n_features = trainX.shape[2]
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)
#%% Build the model for healthy under different input
#####train##############
###LSTM

# epoch and batch size setting
epochs = 500
# batch size for stateful lstm
#validation_percent=0.25
#batch_size_stateful=get_batch_size(trainX, testX, validation_percent)
#possible_batch=find_all_factors(batch_size_stateful)
# batch size for stateless
batch_size=32
# create model

# stateless lstm
model = stateless_model(epochs, trainy.shape[1])

# Bidrectional lstm
model = bidirectional_model(epochs, trainy.shape[1])

'''
#CNN LSTM
model = cnnlstm_model(epochs, trainy.shape[1], time_steps)
'''
'''
# convlstm
model = convlstm_model(epochs,trainy.shape[1],n_seq,n_steps)
'''

# calling callback functions
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
reset_states=ResetStatesCallback()
trainingStopCallback = haltCallback()
lrate = LearningRateScheduler(step_decay)
loss_history = LossHistory()
callbacks_list = [reset_states]
# fit model
#model.summary()
history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, 
                    validation_split=0.2, verbose=1, shuffle=True)


'''
# Hyperparameter tuning
# random search
model = KerasRegressor(build_fn=create_vallina_model, verbose=1, epochs=500)
# define the grid search parameters
#layers=[[8], [8, 4], [8, 8, 4]]
nodes=[32,64,128,256]
layernorm=[True,False]
batch_size = [16,32,64,128]
params = dict(nodes=nodes,layernorm=layernorm,batch_size=batch_size)
random = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='neg_mean_absolute_error', 
                    n_jobs=1)
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
# for random search
'''
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
# demonstrate prediction
target = trainy 
t1 = np.linspace(0, len(target), len(target), endpoint=False)
t2 = np.linspace(0, len(train_predict), len(train_predict), endpoint=False)
fig = plt.figure(figsize=(90,36))
ax1 = plt.subplot(211)
ax1.plot(t1, target, label='observed', linestyle='-', color = 'k', linewidth=7.5)
ax1.plot(t2, train_predict, label='forecast', linestyle=':', color = 'b', linewidth=7.5)
ax1.set_xlim(0,len(target))
ax1.set_ylabel("Absolute acceleration (m/$s^2$)",fontsize=80)
ax1.set_title("Obseved/Predicted Unmeasured Floor Response", fontsize=125)
ax1.legend(loc=1,prop={"size":80})
ax1.spines['left'].set_linewidth(5)
ax1.spines['right'].set_linewidth(5)
ax1.spines['bottom'].set_linewidth(5)
ax1.spines['top'].set_linewidth(5)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
ax2 = plt.subplot(212, sharex=ax1)
# plot the error on the testing set
train_abs_error = np.zeros(train_predict.shape)
for i in range(len(trainy)):
    train_abs_error[i] = np.abs((train_predict[i] - trainy[i]))
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
#%% health(if kobe skip this cell)
# summarize model.
model.summary()
# load test data 
'''
#10% noises
test_kobe = np.loadtxt('./10%_kobe_responses_1_8DOF_500Hz_a.dat')
test_kobe = test_kobe[2000:6000,:] 
plt.plot(test_kobe)     # from the plot, we can guess the acceleration units should be g
plt.show()

test_northridge = np.loadtxt('./10%_northridge_responses_1_8DOF_500Hz_a.dat')
test_northridge=test_northridge[250:3500]
plt.plot(test_northridge)     
plt.show()

test_turkey = np.loadtxt('./10%_turkey_responses_1_8DOF_500Hz_a.dat')
test_turkey=test_turkey[3500:6000,:]
plt.plot(test_turkey)    
plt.show()

test_italy = np.loadtxt('10%_09italy_responses_1_8DOF_500Hz_a.dat')
test_italy=test_italy[3500:4750]
plt.plot(test_italy)
plt.show()

test_hualien = np.loadtxt('10%_180206hualien_responses_1_8DOF_500Hz_a.dat')
test_hualien=test_hualien[2700:5400]
plt.plot(test_hualien)
plt.show()
'''
# 5% noises
test_kobe = np.loadtxt('./05%_kobe_responses_1_8DOF_500Hz_a.dat')
test_kobe = test_kobe[2000:6000,:] 

test_northridge = np.loadtxt('./05%_northridge_responses_1_8DOF_500Hz_a.dat')
test_northridge=test_northridge[250:4000,:]

test_turkey = np.loadtxt('./05%_turkey_responses_1_8DOF_500Hz_a.dat')
test_turkey=test_turkey[3500:6000,:]

test_italy = np.loadtxt('05%_09italy_responses_1_8DOF_500Hz_a.dat')
test_italy=test_italy[3000:6000,:]

test_hualien = np.loadtxt('05%_hualien2018_responses_1_8DOF_500Hz_a.dat')
test_hualien=test_hualien[2500:5000,:]

# test data
#test1=test_kobe
#test1=test_northridge
test1 = test_turkey
#test1 = test_italy
#test1 = test_hualien
plt.plot(test1)    
plt.show()
# test1 = test1 * 9.81  #convert to m/s2 
print(test1.shape)

test_data = test1.T
for i in range(test_data.shape[0]):
    test_temp=decrease_noises(test_data[i], time_steps=5)
    test_data[i,:len(test_temp)]=test_temp
    
test_input = test_data[8,:]
first_test = test_data[7,:]
second_test = test_data[6,:]
third_test = test_data[5,:]
fourth_test = test_data[4,:]
fifth_test = test_data[3,:]
sixth_test = test_data[2,:]
seventh_test = test_data[1,:]
eighth_test= test_data[0,:]
#%% Normalization of test data
# reshape 
test_input = test_input.reshape((len(test_input), 1))
first_test = first_test.reshape((len(first_test),1))
second_test = second_test.reshape((len(second_test),1))
third_test = third_test.reshape((len(third_test),1))
fourth_test = fourth_test.reshape(len(fourth_test),1)
fifth_test = fifth_test.reshape((len(fifth_test),1))
sixth_test = sixth_test.reshape((len(sixth_test),1))
seventh_test = seventh_test.reshape((len(seventh_test),1))
eighth_test = eighth_test.reshape((len(eighth_test), 1))
# concat to a train and test matrix for normalization preprocessing
test_matrix=np.hstack((eighth_test,seventh_test,sixth_test,fifth_test,
                            fourth_test,third_test,second_test,first_test,test_input))

# normalization(scaling data in a range of -1~1)
test_matrix=test_matrix/np.max(abs(test_matrix))
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
test = np.hstack((test_input, first_test, fifth_test, eighth_test, third_test)) # vary in accordance with d.o.f. of training dataset
# split sequences by time steps
testX, testy = split_multisequence(test, n_steps)
testy = testy.reshape((-1,1))
# concatenate along columns to assemble trainX, testX
testX_health = testX
testy_health = testy
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)
print('testX:',testX.shape)
print('testy:',testy.shape)

#%% TEST(HEALTH)
# predict test data
# test
yhat = model.predict(testX_health, verbose=0)
print(yhat)
print(yhat.shape)
print('test MSE:',mean_squared_error(testy, yhat))
# demonstrate test prediction
h=1/500
target = testy
t1 = np.linspace(0, len(target), len(target), endpoint=False)
t2 = np.linspace(0, len(yhat), len(yhat), endpoint=False)
fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot(211)
ax1.plot(t1*h, target, label='observed', linestyle='-', color = 'k', linewidth=7.5)
ax1.plot(t2*h, yhat, label='forecast', linestyle=':', color = 'b', linewidth=7.5)
ax1.set_xlim(0,len(target))
ax1.set_ylabel("Absolute Acceleration Response(m/$s^2$)",fontsize=80)
ax1.set_title("Obseved/Predicted Base Excitation", fontsize=125)
ax1.legend(loc=1,prop={"size":80})
ax1.spines['left'].set_linewidth(5)
ax1.spines['right'].set_linewidth(5)
ax1.spines['bottom'].set_linewidth(5)
ax1.spines['top'].set_linewidth(5)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
ax2 = plt.subplot(212, sharex=ax1)
# plot the error on the testing set
error = np.zeros(yhat.shape)
for i in range(len(testy)):
    error[i] = np.abs((yhat[i] - testy[i]))
ax2.plot(error, linewidth=7.5)
ax2.set_ylim(0,error.max())
ax2.set_xlabel("Number of samples",fontsize=80)
ax2.set_ylabel("Absolute error",fontsize=80)
ax2.spines['left'].set_linewidth(5)
ax2.spines['right'].set_linewidth(5)
ax2.spines['bottom'].set_linewidth(5)
ax2.spines['top'].set_linewidth(5)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.show()

#%% save results
model.save("lstm_158pred3.h5")
print("Saved model to disk")
train_error=train_predict-trainy
error=yhat-testy
np.save('train_error_lstm',train_error)
np.save('error_lstm',error)