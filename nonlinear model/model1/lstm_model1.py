# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:50:32 2020

@author: CSHuangLab
"""


# %% import modules
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, PReLU, Dropout, Masking, Flatten
from tensorflow.keras.layers import LayerNormalization, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, ConvLSTM2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import scipy
import seaborn as sns
import pandas as pd
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
# Stateleess LSTM
def stateless_model(epochs, output_dim):
    #adam = K.optimizers.Adam(learning_rate=0.003)
    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
    model.add(LSTM(32, activation='tanh',# kernel_initializer='he_uniform', 
                   #kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #bias_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   #recurrent_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
                   return_sequences=False, stateful=False))
    #model.add(LayerNormalization())
    #model.add(PReLU())
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1))
    #model.add(LSTM(64, activation='tanh',
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
# Bidirectional LSTM
def bidirectional_model(epochs,output_dim):
    #adam = K.optimizers.Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
    model.add(Bidirectional(LSTM(32, activation='tanh',
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
            model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
            model.add(LSTM(layers[0], activation='tanh',return_sequences=False))
            model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1)
        elif len(layers)==2: 
            model.trainable = False
            model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
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
            model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
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
        model.add(Dense(8))
        adam = K.optimizers.Adam(learning_rate=learn_rate)
        model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    else:
        if len(layers)==1:
            model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
            model.add(LSTM(layers[0], activation='tanh', return_sequences=False))
            #model.add(LayerNormalization())
            #model.add(BatchNormalization())
            #model.add(PReLU())
            #model.add(Dropout(0.1)
        elif len(layers)==2: 
            model.trainable = False
            model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
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
            model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
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
        model.add(Dense(8))
        adam = K.optimizers.Adam(learning_rate=learn_rate)
        model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model
def create_vallina_model(nodes, layernorm):   
    model = Sequential()
    if layernorm:
        model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
        model.add(LSTM(nodes, activation='tanh',return_sequences=False))
        model.add(LayerNormalization())
        #model.add(BatchNormalization())
        #model.add(PReLU())
        #model.add(Dropout(0.1)
        model.add(Dense(8))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
        model.add(LSTM(nodes, activation='tanh', return_sequences=False))
        #model.add(LayerNormalization())
        #model.add(BatchNormalization())
        #model.add(PReLU())
        #model.add(Dropout(0.1)
        model.add(Dense(8))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
# determine learning rate
def create_vallina_lstm(lr):   
    model = Sequential()
    #model.add(Masking(mask_value=special_value, input_shape=(n_steps, n_features)))
    model.add(LSTM(32, activation='tanh',return_sequences=False, input_shape=(n_steps, n_features)))
    #model.add(LayerNormalization())
    #model.add(BatchNormalization())
    #model.add(PReLU())
    #model.add(Dropout(0.1)
    model.add(Dense(8))
    adam = K.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model
#%% load training data
train_chichi=pd.read_excel('CHICHI.xlsx')
train_chichi = train_chichi.drop([0,1])
train_chichi = train_chichi.iloc[:, 1:10]
train_chichi = train_chichi.reset_index(drop=True)
train_chichi = train_chichi.astype('float')
train_chichi = train_chichi.to_numpy(dtype=float)
train_chichi = train_chichi/1000
train_chichi = train_chichi[5000:15000]

train_chiayi=pd.read_excel('1999CHIAYI.xlsx')
train_chiayi = train_chiayi.drop([0,1])
train_chiayi = train_chiayi.iloc[:, 1:10]
train_chiayi = train_chiayi.reset_index(drop=True)
train_chiayi = train_chiayi.astype('float')
train_chiayi = train_chiayi.to_numpy(dtype=float)
train_chiayi = train_chiayi/1000
train_chiayi = train_chiayi[2000:6250]

train_hualien331=pd.read_excel('2002HUALIEN.xlsx')
train_hualien331 = train_hualien331.drop([0,1])
train_hualien331 = train_hualien331.iloc[:, 1:10]
train_hualien331 = train_hualien331.reset_index(drop=True)
train_hualien331 = train_hualien331.astype('float')
train_hualien331 = train_hualien331.to_numpy(dtype=float)
train_hualien331 = train_hualien331/1000
train_hualien331 = train_hualien331[4000:12000]

train_meinung=pd.read_excel('2016MEINUNG.xlsx')
train_meinung = train_meinung.drop([0,1])
train_meinung = train_meinung.iloc[:, 1:10]
train_meinung = train_meinung.reset_index(drop=True)
train_meinung = train_meinung.astype('float')
train_meinung = train_meinung.to_numpy(dtype=float)
train_meinung = train_meinung/1000
train_meinung = train_meinung[3750:12000]

train_california20=pd.read_excel('2020CALIFORNIA.xlsx')
train_california20 = train_california20.drop([0,1])
train_california20 = train_california20.iloc[:, 1:10]
train_california20 = train_california20.reset_index(drop=True)
train_california20 = train_california20.astype('float')
train_california20 = train_california20.to_numpy(dtype=float)
train_california20 = train_california20/1000
train_california20 = train_california20[2750:7500]

train_yilan=pd.read_excel('20190808YILAN.xlsx')
train_yilan = train_yilan.drop([0,1])
train_yilan = train_yilan.iloc[:, 1:10]
train_yilan = train_yilan.reset_index(drop=True)
train_yilan = train_yilan.astype('float')
train_yilan = train_yilan.to_numpy(dtype=float)
train_yilan = train_yilan/1000
train_yilan = train_yilan[3750:7500]

train_amberley=pd.read_excel('AMBERLEY.xlsx')
train_amberley = train_amberley.drop([0,1])
train_amberley = train_amberley.iloc[:, 1:10]
train_amberley = train_amberley.reset_index(drop=True)
train_amberley = train_amberley.astype('float')
train_amberley = train_amberley.to_numpy(dtype=float)
train_amberley = train_amberley/1000
train_amberley = train_amberley[10000:40000]

train_athens=pd.read_excel('ATHENS.xlsx')
train_athens = train_athens.drop([0,1])
train_athens = train_athens.iloc[:, 1:10]
train_athens = train_athens.reset_index(drop=True)
train_athens = train_athens.astype('float')
train_athens = train_athens.to_numpy(dtype=float)
train_athens = train_athens/1000
train_athens = train_athens[500:6250]

train_corralitos=pd.read_excel('CORRALITOS.xlsx')
train_corralitos = train_corralitos.drop([0,1])
train_corralitos = train_corralitos.iloc[:, 1:10]
train_corralitos = train_corralitos.reset_index(drop=True)
train_corralitos = train_corralitos.astype('float')
train_corralitos = train_corralitos.to_numpy(dtype=float)
train_corralitos = train_corralitos/1000
train_corralitos = train_corralitos[:5000]

train_elcentro=pd.read_excel('ELCENTRO.xlsx')
train_elcentro = train_elcentro.drop([0,1])
train_elcentro = train_elcentro.iloc[:, 1:10]
train_elcentro = train_elcentro.reset_index(drop=True)
train_elcentro = train_elcentro.astype('float')
train_elcentro = train_elcentro.to_numpy(dtype=float)
train_elcentro = train_elcentro/1000
train_elcentro = train_elcentro[:7500]

train_emilyville=pd.read_excel('EMILYVILLE.xlsx')
train_emilyville = train_emilyville.drop([0,1])
train_emilyville = train_emilyville.iloc[:, 1:10]
train_emilyville = train_emilyville.reset_index(drop=True)
train_emilyville = train_emilyville.astype('float')
train_emilyville = train_emilyville.to_numpy(dtype=float)
train_emilyville = train_emilyville/1000
train_emilyville = train_emilyville[:6000]

train_friulli=pd.read_excel('FRIULLI.xlsx')
train_friulli = train_friulli.drop([0,1])
train_friulli = train_friulli.iloc[:, 1:10]
train_friulli = train_friulli.reset_index(drop=True)
train_friulli = train_friulli.astype('float')
train_friulli = train_friulli.to_numpy(dtype=float)
train_friulli = train_friulli/1000
train_friulli = train_friulli[:4000]

train_hollister=pd.read_excel('HOLLISTER.xlsx')
train_hollister = train_hollister.drop([0,1])
train_hollister = train_hollister.iloc[:, 1:10]
train_hollister = train_hollister.reset_index(drop=True)
train_hollister = train_hollister.astype('float')
train_hollister = train_hollister.to_numpy(dtype=float)
train_hollister = train_hollister/1000
train_hollister = train_hollister[:5000]

train_jp311=pd.read_excel('JP311.xlsx')
train_jp311 = train_jp311.drop([0,1])
train_jp311 = train_jp311.iloc[:, 1:10]
train_jp311 = train_jp311.reset_index(drop=True)
train_jp311 = train_jp311.astype('float')
train_jp311 = train_jp311.to_numpy(dtype=float)
train_jp311 = train_jp311/1000
train_jp311 = train_jp311[5000:25000]

train_kocaeli=pd.read_excel('KOCAELI.xlsx')
train_kocaeli = train_kocaeli.drop([0,1])
train_kocaeli = train_kocaeli.iloc[:, 1:10]
train_kocaeli = train_kocaeli.reset_index(drop=True)
train_kocaeli = train_kocaeli.astype('float')
train_kocaeli = train_kocaeli.to_numpy(dtype=float)
train_kocaeli = train_kocaeli/1000
train_kocaeli = train_kocaeli[:7500]

train_2011turkey=pd.read_excel('2011TURKEY.xlsx')
train_2011turkey = train_2011turkey.drop([0,1])
train_2011turkey = train_2011turkey.iloc[:, 1:10]
train_2011turkey = train_2011turkey.reset_index(drop=True)
train_2011turkey = train_2011turkey.astype('float')
train_2011turkey = train_2011turkey.to_numpy(dtype=float)
train_2011turkey = train_2011turkey/1000
train_2011turkey = train_2011turkey[2500:7500]

train_northridge=pd.read_excel('NORTHRIDGE.xlsx')
train_northridge = train_northridge.drop([0,1])
train_northridge = train_northridge.iloc[:, 1:10]
train_northridge = train_northridge.reset_index(drop=True)
train_northridge = train_northridge.astype('float')
train_northridge = train_northridge.to_numpy(dtype=float)
train_northridge = train_northridge/1000
train_northridge = train_northridge[:7500]

train_rome=pd.read_excel('ROME.xlsx')
train_rome = train_rome.drop([0,1])
train_rome = train_rome.iloc[:, 1:10]
train_rome = train_rome.reset_index(drop=True)
train_rome = train_rome.astype('float')
train_rome = train_rome.to_numpy(dtype=float)
train_rome = train_rome/1000
train_rome = train_rome[1000:5000]

train_sakaria=pd.read_excel('SAKARIA.xlsx')
train_sakaria = train_sakaria.drop([0,1])
train_sakaria = train_sakaria.iloc[:, 1:10]
train_sakaria = train_sakaria.reset_index(drop=True)
train_sakaria = train_sakaria.astype('float')
train_sakaria = train_sakaria.to_numpy(dtype=float)
train_sakaria = train_sakaria/1000
train_sakaria = train_sakaria[:5000]

train_norica=pd.read_excel('NORICA.xlsx')
train_norica = train_norica.drop([0,1])
train_norica = train_norica.iloc[:, 1:10]
train_norica = train_norica.reset_index(drop=True)
train_norica = train_norica.astype('float')
train_norica = train_norica.to_numpy(dtype=float)
train_norica = train_norica/1000
train_norica = train_norica[2000:6000]

#%% absolute acc
for i in range(train_chichi.shape[1]-1):
    train_chichi[:,i]=train_chichi[:,i]+train_chichi[:,8]
    train_norica[:,i]=train_norica[:,i]+train_norica[:,8]
    train_hualien331[:,i]=train_hualien331[:,i]+train_hualien331[:,8]
    train_2011turkey[:,i]=train_2011turkey[:,i]+train_2011turkey[:,8]
    train_hollister[:,i]=train_hollister[:,i]+train_hollister[:,8]
    train_jp311[:,i]=train_jp311[:,i]+train_jp311[:,8]
    train_friulli[:,i]=train_friulli[:,i]+train_friulli[:,8]
    train_amberley[:,i]=train_amberley[:,i]+train_amberley[:,8]
    train_emilyville[:,i]=train_emilyville[:,i]+train_emilyville[:,8]
    train_corralitos[:,i]=train_corralitos[:,i]+train_corralitos[:,8]
    train_california20[:,i]=train_california20[:,i]+train_california20[:,8]
    train_elcentro[:,i]=train_elcentro[:,i]+train_elcentro[:,8]
    train_chiayi[:,i]=train_chiayi[:,i]+train_chiayi[:,8]
    train_yilan[:,i]=train_yilan[:,i]+train_yilan[:,8]
    train_rome[:,i]=train_rome[:,i]+train_rome[:,8]
    train_sakaria[:,i]=train_sakaria[:,i]+train_sakaria[:,8]
    train_athens[:,i]=train_athens[:,i]+train_athens[:,8]
    train_kocaeli[:,i]=train_kocaeli[:,i]+train_kocaeli[:,8]
    train_meinung[:,i]=train_meinung[:,i]+train_meinung[:,8]
    train_northridge[:,i]=train_northridge[:,i]+train_northridge[:,8]


#%% normalization(scaling data in a range of -1~1)
train_matrix_chichi=train_chichi/np.max(abs(train_chichi))
train_matrix_chiayi=train_chiayi/np.max(abs(train_chiayi))
train_matrix_hualien331=train_hualien331/np.max(abs(train_hualien331))
train_matrix_meinung=train_meinung/np.max(abs(train_meinung))
train_matrix_california20=train_california20/np.max(abs(train_california20))
train_matrix_yilan=train_yilan/np.max(abs(train_yilan))
train_matrix_amberley=train_amberley/np.max(abs(train_amberley))
train_matrix_athens=train_athens/np.max(abs(train_athens))
train_matrix_corralitos=train_corralitos/np.max(abs(train_corralitos))
train_matrix_elcentro=train_elcentro/np.max(abs(train_elcentro))
train_matrix_emilyville=train_emilyville/np.max(abs(train_emilyville))
train_matrix_friulli=train_friulli/np.max(abs(train_friulli))
train_matrix_hollister=train_hollister/np.max(abs(train_hollister))
train_matrix_jp311=train_jp311/np.max(abs(train_jp311))
train_matrix_kocaeli=train_kocaeli/np.max(abs(train_kocaeli))
train_matrix_2011turkey=train_2011turkey/np.max(abs(train_2011turkey))
train_matrix_northridge=train_northridge/np.max(abs(train_northridge))
train_matrix_rome=train_rome/np.max(abs(train_rome))
train_matrix_sakaria=train_sakaria/np.max(abs(train_sakaria))
train_matrix_norica=train_norica/np.max(abs(train_norica))

chichi_input = train_matrix_chichi[:,8].reshape((-1,1))
first_chichi = train_matrix_chichi[:,7].reshape((-1,1))
second_chichi = train_matrix_chichi[:,6].reshape((-1,1))
third_chichi = train_matrix_chichi[:,5].reshape((-1,1))
fourth_chichi = train_matrix_chichi[:,4].reshape((-1,1))
fifth_chichi = train_matrix_chichi[:,3].reshape((-1,1))
sixth_chichi = train_matrix_chichi[:,2].reshape((-1,1))
seventh_chichi = train_matrix_chichi[:,1].reshape((-1,1))
eighth_chichi = train_matrix_chichi[:,0].reshape((-1,1))

chiayi_input = train_matrix_chiayi[:,8].reshape((-1,1))
first_chiayi = train_matrix_chiayi[:,7].reshape((-1,1))
second_chiayi = train_matrix_chiayi[:,6].reshape((-1,1))
third_chiayi = train_matrix_chiayi[:,5].reshape((-1,1))
fourth_chiayi = train_matrix_chiayi[:,4].reshape((-1,1))
fifth_chiayi = train_matrix_chiayi[:,3].reshape((-1,1))
sixth_chiayi = train_matrix_chiayi[:,2].reshape((-1,1))
seventh_chiayi = train_matrix_chiayi[:,1].reshape((-1,1))
eighth_chiayi = train_matrix_chiayi[:,0].reshape((-1,1))

hualien331_input = train_matrix_hualien331[:,8].reshape((-1,1))
first_hualien331 = train_matrix_hualien331[:,7].reshape((-1,1))
second_hualien331 = train_matrix_hualien331[:,6].reshape((-1,1))
third_hualien331 = train_matrix_hualien331[:,5].reshape((-1,1))
fourth_hualien331 = train_matrix_hualien331[:,4].reshape((-1,1))
fifth_hualien331 = train_matrix_hualien331[:,3].reshape((-1,1))
sixth_hualien331 = train_matrix_hualien331[:,2].reshape((-1,1))
seventh_hualien331 = train_matrix_hualien331[:,1].reshape((-1,1))
eighth_hualien331 = train_matrix_hualien331[:,0].reshape((-1,1))

meinung_input = train_matrix_meinung[:,8].reshape((-1,1))
first_meinung = train_matrix_meinung[:,7].reshape((-1,1))
second_meinung = train_matrix_meinung[:,6].reshape((-1,1))
third_meinung = train_matrix_meinung[:,5].reshape((-1,1))
fourth_meinung = train_matrix_meinung[:,4].reshape((-1,1))
fifth_meinung = train_matrix_meinung[:,3].reshape((-1,1))
sixth_meinung = train_matrix_meinung[:,2].reshape((-1,1))
seventh_meinung = train_matrix_meinung[:,1].reshape((-1,1))
eighth_meinung = train_matrix_meinung[:,0].reshape((-1,1))

california20_input = train_matrix_california20[:,8].reshape((-1,1))
first_california20 = train_matrix_california20[:,7].reshape((-1,1))
second_california20 = train_matrix_california20[:,6].reshape((-1,1))
third_california20 = train_matrix_california20[:,5].reshape((-1,1))
fourth_california20 = train_matrix_california20[:,4].reshape((-1,1))
fifth_california20  = train_matrix_california20[:,3].reshape((-1,1))
sixth_california20 = train_matrix_california20[:,2].reshape((-1,1))
seventh_california20 = train_matrix_california20[:,1].reshape((-1,1))
eighth_california20 = train_matrix_california20[:,0].reshape((-1,1))

yilan_input = train_matrix_yilan[:,8].reshape((-1,1))
first_yilan = train_matrix_yilan[:,7].reshape((-1,1))
second_yilan = train_matrix_yilan[:,6].reshape((-1,1))
third_yilan = train_matrix_yilan[:,5].reshape((-1,1))
fourth_yilan = train_matrix_yilan[:,4].reshape((-1,1))
fifth_yilan = train_matrix_yilan[:,3].reshape((-1,1))
sixth_yilan = train_matrix_yilan[:,2].reshape((-1,1))
seventh_yilan = train_matrix_yilan[:,1].reshape((-1,1))
eighth_yilan = train_matrix_yilan[:,0].reshape((-1,1))

amberley_input = train_matrix_amberley[:,8].reshape((-1,1))
first_amberley = train_matrix_amberley[:,7].reshape((-1,1))
second_amberley = train_matrix_amberley[:,6].reshape((-1,1))
third_amberley = train_matrix_amberley[:,5].reshape((-1,1))
fourth_amberley = train_matrix_amberley[:,4].reshape((-1,1))
fifth_amberley  = train_matrix_amberley[:,3].reshape((-1,1))
sixth_amberley = train_matrix_amberley[:,2].reshape((-1,1))
seventh_amberley = train_matrix_amberley[:,1].reshape((-1,1))
eighth_amberley = train_matrix_amberley[:,0].reshape((-1,1))

athens_input = train_matrix_athens[:,8].reshape((-1,1))
first_athens = train_matrix_athens[:,7].reshape((-1,1))
second_athens = train_matrix_athens[:,6].reshape((-1,1))
third_athens = train_matrix_athens[:,5].reshape((-1,1))
fourth_athens = train_matrix_athens[:,4].reshape((-1,1))
fifth_athens = train_matrix_athens[:,3].reshape((-1,1))
sixth_athens = train_matrix_athens[:,2].reshape((-1,1))
seventh_athens = train_matrix_athens[:,1].reshape((-1,1))
eighth_athens = train_matrix_athens[:,0].reshape((-1,1))

corralitos_input = train_matrix_corralitos[:,8].reshape((-1,1))
first_corralitos = train_matrix_corralitos[:,7].reshape((-1,1))
second_corralitos = train_matrix_corralitos[:,6].reshape((-1,1))
third_corralitos= train_matrix_corralitos[:,5].reshape((-1,1))
fourth_corralitos = train_matrix_corralitos[:,4].reshape((-1,1))
fifth_corralitos = train_matrix_corralitos[:,3].reshape((-1,1))
sixth_corralitos = train_matrix_corralitos[:,2].reshape((-1,1))
seventh_corralitos = train_matrix_corralitos[:,1].reshape((-1,1))
eighth_corralitos = train_matrix_corralitos[:,0].reshape((-1,1))

elcentro_input = train_matrix_elcentro[:,8].reshape((-1,1))
first_elcentro = train_matrix_elcentro[:,7].reshape((-1,1))
second_elcentro = train_matrix_elcentro[:,6].reshape((-1,1))
third_elcentro = train_matrix_elcentro[:,5].reshape((-1,1))
fourth_elcentro = train_matrix_elcentro[:,4].reshape((-1,1))
fifth_elcentro = train_matrix_elcentro[:,3].reshape((-1,1))
sixth_elcentro = train_matrix_elcentro[:,2].reshape((-1,1))
seventh_elcentro = train_matrix_elcentro[:,1].reshape((-1,1))
eighth_elcentro = train_matrix_elcentro[:,0].reshape((-1,1))

emilyville_input = train_matrix_emilyville[:,8].reshape((-1,1))
first_emilyville = train_matrix_emilyville[:,7].reshape((-1,1))
second_emilyville = train_matrix_emilyville[:,6].reshape((-1,1))
third_emilyville = train_matrix_emilyville[:,5].reshape((-1,1))
fourth_emilyville = train_matrix_emilyville[:,4].reshape((-1,1))
fifth_emilyville = train_matrix_emilyville[:,3].reshape((-1,1))
sixth_emilyville = train_matrix_emilyville[:,2].reshape((-1,1))
seventh_emilyville = train_matrix_emilyville[:,1].reshape((-1,1))
eighth_emilyville = train_matrix_emilyville[:,0].reshape((-1,1))

friulli_input = train_matrix_friulli[:,8].reshape((-1,1))
first_friulli = train_matrix_friulli[:,7].reshape((-1,1))
second_friulli = train_matrix_friulli[:,6].reshape((-1,1))
third_friulli = train_matrix_friulli[:,5].reshape((-1,1))
fourth_friulli = train_matrix_friulli[:,4].reshape((-1,1))
fifth_friulli  = train_matrix_friulli[:,3].reshape((-1,1))
sixth_friulli = train_matrix_friulli[:,2].reshape((-1,1))
seventh_friulli = train_matrix_friulli[:,1].reshape((-1,1))
eighth_friulli = train_matrix_friulli[:,0].reshape((-1,1))

hollister_input = train_matrix_hollister[:,8].reshape((-1,1))
first_hollister = train_matrix_hollister[:,7].reshape((-1,1))
second_hollister = train_matrix_hollister[:,6].reshape((-1,1))
third_hollister = train_matrix_hollister[:,5].reshape((-1,1))
fourth_hollister = train_matrix_hollister[:,4].reshape((-1,1))
fifth_hollister = train_matrix_hollister[:,3].reshape((-1,1))
sixth_hollister = train_matrix_hollister[:,2].reshape((-1,1))
seventh_hollister = train_matrix_hollister[:,1].reshape((-1,1))
eighth_hollister = train_matrix_hollister[:,0].reshape((-1,1))

jp311_input = train_matrix_jp311[:,8].reshape((-1,1))
first_jp311 = train_matrix_jp311[:,7].reshape((-1,1))
second_jp311 = train_matrix_jp311[:,6].reshape((-1,1))
third_jp311 = train_matrix_jp311[:,5].reshape((-1,1))
fourth_jp311 = train_matrix_jp311[:,4].reshape((-1,1))
fifth_jp311 = train_matrix_jp311[:,3].reshape((-1,1))
sixth_jp311 = train_matrix_jp311[:,2].reshape((-1,1))
seventh_jp311 = train_matrix_jp311[:,1].reshape((-1,1))
eighth_jp311 = train_matrix_jp311[:,0].reshape((-1,1))

kocaeli_input = train_matrix_kocaeli[:,8].reshape((-1,1))
first_kocaeli = train_matrix_kocaeli[:,7].reshape((-1,1))
second_kocaeli = train_matrix_kocaeli[:,6].reshape((-1,1))
third_kocaeli = train_matrix_kocaeli[:,5].reshape((-1,1))
fourth_kocaeli = train_matrix_kocaeli[:,4].reshape((-1,1))
fifth_kocaeli = train_matrix_kocaeli[:,3].reshape((-1,1))
sixth_kocaeli = train_matrix_kocaeli[:,2].reshape((-1,1))
seventh_kocaeli = train_matrix_kocaeli[:,1].reshape((-1,1))
eighth_kocaeli = train_matrix_kocaeli[:,0].reshape((-1,1))

turkey_input = train_matrix_2011turkey[:,8].reshape((-1,1))
first_turkey = train_matrix_2011turkey[:,7].reshape((-1,1))
second_turkey = train_matrix_2011turkey[:,6].reshape((-1,1))
third_turkey = train_matrix_2011turkey[:,5].reshape((-1,1))
fourth_turkey = train_matrix_2011turkey[:,4].reshape((-1,1))
fifth_turkey  = train_matrix_2011turkey[:,3].reshape((-1,1))
sixth_turkey = train_matrix_2011turkey[:,2].reshape((-1,1))
seventh_turkey = train_matrix_2011turkey[:,1].reshape((-1,1))
eighth_turkey = train_matrix_2011turkey[:,0].reshape((-1,1))

northridge_input = train_matrix_northridge[:,8].reshape((-1,1))
first_northridge = train_matrix_northridge[:,7].reshape((-1,1))
second_northridge = train_matrix_northridge[:,6].reshape((-1,1))
third_northridge = train_matrix_northridge[:,5].reshape((-1,1))
fourth_northridge = train_matrix_northridge[:,4].reshape((-1,1))
fifth_northridge = train_matrix_northridge[:,3].reshape((-1,1))
sixth_northridge = train_matrix_northridge[:,2].reshape((-1,1))
seventh_northridge = train_matrix_northridge[:,1].reshape((-1,1))
eighth_northridge = train_matrix_northridge[:,0].reshape((-1,1))

rome_input = train_matrix_rome[:,8].reshape((-1,1))
first_rome = train_matrix_rome[:,7].reshape((-1,1))
second_rome = train_matrix_rome[:,6].reshape((-1,1))
third_rome = train_matrix_rome[:,5].reshape((-1,1))
fourth_rome = train_matrix_rome[:,4].reshape((-1,1))
fifth_rome = train_matrix_rome[:,3].reshape((-1,1))
sixth_rome = train_matrix_rome[:,2].reshape((-1,1))
seventh_rome = train_matrix_rome[:,1].reshape((-1,1))
eighth_rome = train_matrix_rome[:,0].reshape((-1,1))

sakaria_input = train_matrix_sakaria[:,8].reshape((-1,1))
first_sakaria = train_matrix_sakaria[:,7].reshape((-1,1))
second_sakaria = train_matrix_sakaria[:,6].reshape((-1,1))
third_sakaria = train_matrix_sakaria[:,5].reshape((-1,1))
fourth_sakaria = train_matrix_sakaria[:,4].reshape((-1,1))
fifth_sakaria = train_matrix_sakaria[:,3].reshape((-1,1))
sixth_sakaria = train_matrix_sakaria[:,2].reshape((-1,1))
seventh_sakaria = train_matrix_sakaria[:,1].reshape((-1,1))
eighth_sakaria = train_matrix_sakaria[:,0].reshape((-1,1))

norica_input = train_matrix_norica[:,8].reshape((-1,1))
first_norica = train_matrix_norica[:,7].reshape((-1,1))
second_norica = train_matrix_norica[:,6].reshape((-1,1))
third_norica = train_matrix_norica[:,5].reshape((-1,1))
fourth_norica = train_matrix_norica[:,4].reshape((-1,1))
fifth_norica = train_matrix_norica[:,3].reshape((-1,1))
sixth_norica = train_matrix_norica[:,2].reshape((-1,1))
seventh_norica = train_matrix_norica[:,1].reshape((-1,1))
eighth_norica = train_matrix_norica[:,0].reshape((-1,1))
# %% split the data (time steps)
# horizontally stack columns
chichi = np.hstack((first_chichi, fourth_chichi, eighth_chichi))
chiayi=np.hstack((first_chiayi,fourth_chiayi, eighth_chiayi))
hualien331=np.hstack((first_hualien331,fourth_hualien331,eighth_hualien331))
meinung=np.hstack((first_meinung,fourth_meinung,eighth_meinung))
california20=np.hstack((first_california20,fourth_california20,eighth_california20))
yilan=np.hstack((first_yilan,fourth_yilan,eighth_yilan))
amberley=np.hstack((first_amberley,fourth_amberley,eighth_amberley))
athens=np.hstack((first_athens,fourth_athens,eighth_athens))
corralitos = np.hstack((first_corralitos,fourth_corralitos,eighth_corralitos))
elcentro = np.hstack((first_elcentro,fourth_elcentro,eighth_elcentro))
emilyville=np.hstack((first_emilyville,fourth_emilyville,eighth_emilyville))
friulli=np.hstack((first_friulli,fourth_friulli,eighth_friulli))
hollister=np.hstack((first_hollister,fourth_hollister,eighth_hollister))
jp311=np.hstack((first_jp311,fourth_jp311,eighth_jp311))
kocaeli=np.hstack((first_kocaeli,fourth_kocaeli,eighth_kocaeli))
turkey=np.hstack((first_turkey,fourth_turkey,eighth_turkey))
northridge=np.hstack((first_northridge,fourth_northridge,eighth_northridge))
rome=np.hstack((first_rome,fourth_rome,eighth_rome))
sakaria=np.hstack((first_sakaria,fourth_sakaria,eighth_sakaria))
norica = np.hstack((first_norica,fourth_norica,eighth_norica))
# split sequences by time steps
n_steps = 10
chichiX1, chichi_y = split_multisequence(chichi, n_steps-1)
chiayiX1, chiayi_y = split_multisequence(chiayi, n_steps-1)
hualien331X1, hualien331_y = split_multisequence(hualien331, n_steps-1)
meinungX1, meinung_y = split_multisequence(meinung, n_steps-1)
california20X1, california20_y = split_multisequence(california20, n_steps-1)
yilanX1, yilan_y = split_multisequence(yilan, n_steps-1)
amberleyX1, amberley_y = split_multisequence(amberley, n_steps-1)
athensX1, athens_y = split_multisequence(athens, n_steps-1)
corralitosX1, corralitos_y = split_multisequence(corralitos, n_steps-1)
elcentroX1, elcentro_y = split_multisequence(elcentro, n_steps-1)
emilyvilleX1, emilyville_y = split_multisequence(emilyville, n_steps-1)
friulliX1, friulli_y = split_multisequence(friulli, n_steps-1)
hollisterX1, hollister_y = split_multisequence(hollister, n_steps-1)
jp311X1, jp311_y = split_multisequence(jp311, n_steps-1)
kocaeliX1, kocaeli_y = split_multisequence(kocaeli, n_steps-1)
turkeyX1, turkey_y = split_multisequence(turkey, n_steps-1)
northridgeX1, northridge_y = split_multisequence(northridge, n_steps-1)
romeX1, rome_y = split_multisequence(rome, n_steps-1)
sakariaX1, sakaria_y = split_multisequence(sakaria, n_steps-1)
noricaX1, norica_y = split_multisequence(norica, n_steps-1)

chichiX2 = split_sequence(chichi_input, n_steps)
chiayiX2 = split_sequence(chiayi_input, n_steps)
hualien331X2 = split_sequence(hualien331_input, n_steps)
meinungX2 = split_sequence(meinung_input, n_steps)
california20X2 = split_sequence(california20_input, n_steps)
yilanX2 = split_sequence(yilan_input, n_steps)
amberleyX2 = split_sequence(amberley_input, n_steps)
athensX2 = split_sequence(athens_input, n_steps)
corralitosX2 = split_sequence(corralitos_input, n_steps)
elcentroX2 = split_sequence(elcentro_input, n_steps)
emilyvilleX2 = split_sequence(emilyville_input, n_steps)
friulliX2 = split_sequence(friulli_input, n_steps)
hollisterX2 = split_sequence(hollister_input, n_steps)
jp311X2 = split_sequence(jp311_input, n_steps)
kocaeliX2 = split_sequence(kocaeli_input, n_steps)
turkeyX2 = split_sequence(turkey_input, n_steps)
northridgeX2 = split_sequence(northridge_input, n_steps)
romeX2 = split_sequence(rome_input, n_steps)
sakariaX2 = split_sequence(sakaria_input, n_steps)
noricaX2 = split_sequence(norica_input, n_steps)

# 20 cases
trainX1=np.vstack((chichiX1,chiayiX1,hualien331X1,meinungX1,california20X1,yilanX1,
                   amberleyX1,athensX1,corralitosX1,elcentroX1,emilyvilleX1,friulliX1,
                   hollisterX1,jp311X1,kocaeliX1,turkeyX1,northridgeX1,
                   romeX1,sakariaX1,noricaX1))
trainX2=np.vstack((chichiX2,chiayiX2,hualien331X2,meinungX2,california20X2,yilanX2,amberleyX2,athensX2,
                   corralitosX2,elcentroX2,emilyvilleX2,friulliX2,hollisterX2,jp311X2,kocaeliX2,turkeyX2,
                   northridgeX2,romeX2,sakariaX2,noricaX2))
# padding the sequences
special_value = (np.mean(train_chichi)+np.mean(train_corralitos)+np.mean(train_norica)
                 +np.mean(train_elcentro)+np.mean(train_emilyville)+np.mean(train_friulli)
                 +np.mean(train_california20)+np.mean(train_hollister)
                 +np.mean(train_2011turkey)+np.mean(train_jp311)+np.mean(train_amberley)+
                 np.mean(train_hualien331)+np.mean(train_northridge)+np.mean(train_chiayi)+
                 np.mean(train_yilan)+np.mean(train_rome)+np.mean(train_sakaria)
                 +np.mean(train_athens)+np.mean(train_meinung)+np.mean(train_kocaeli))/20

trainy = np.vstack((chichi_y,chiayi_y,hualien331_y,meinung_y,california20_y,yilan_y,
                    amberley_y,athens_y,corralitos_y,elcentro_y,emilyville_y,friulli_y,
                    hollister_y,jp311_y,kocaeli_y,turkey_y,northridge_y,rome_y,sakaria_y,norica_y))

trainXpad = np.full((trainX1.shape[0], n_steps, trainX1.shape[2]), fill_value=special_value)
for s, x in enumerate(trainX1):
    seq_len = x.shape[0]
    trainXpad[s, 0:seq_len, :] = x
# concatenate along the third axis
train_X = np.dstack((trainX2, trainXpad))

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = train_X.shape[2]
trainX = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features)) 

print('trainX:',trainX.shape)
print('trainy:',trainy.shape)
#%% Build the model for healthy under different input
#####train##############
###LSTM
# epoch and batch size setting
epochs = 500
# batch size for stateful lstm
validation_percent=0.25
#batch_size_stateful=get_batch_size(trainX, testX, validation_percent)
#possible_batch=find_all_factors(batch_size_stateful)
# batch size for stateless
batch_size=64
#nodes=int(np.ceil(2/3*(n_features+trainy.shape[1])))
# create model
# stateless lstm
model = stateless_model(epochs, trainy.shape[1])
'''
# Bidrectional lstm
model = bidirectional_model(epochs, trainy.shape[1])
'''
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

model.summary()
history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, 
                    validation_split=0.3, verbose=1, shuffle=False)

'''
# Hyperparameter tuning
# random search
model = KerasRegressor(build_fn=create_vallina_model, verbose=1, epochs=500)
# define the grid search parameters
#layers=[[8], [8, 4], [8, 8, 4]]
nodes=[16,32,64,128]
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
'''
# for random search
model = random_result.best_estimator_.model 
model.summary()
plt.plot(random_result.best_estimator_.model.history.history['loss'], label='train')
plt.plot(random_result.best_estimator_.model.history.history['val_loss'], label='validation')
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
fl=np.array([1,4,8])
for i in range(train_predict.shape[1]):
    pred_fl = train_predict[:,i]
    floor=fl[i]
    target_train=trainy[:,i]
    t1 = np.linspace(0, len(target_train), len(target_train), endpoint=False)
    t2 = np.linspace(n_steps, len(target_train), len(pred_fl), endpoint=False)
    fig = plt.figure(figsize=(90,36))
    ax1 = plt.subplot(211)
    ax1.plot(t1, target_train, label='observed', linestyle='-', color = 'k', linewidth=7.5)
    ax1.plot(t2, pred_fl, label='forecast', linestyle=':', color = 'b', linewidth=7.5)
    ax1.set_xlim(0,len(target_train))
    ax1.set_ylabel("Absolute acceleration (m/$s^2$)",fontsize=80)
    ax1.set_title("%d floor" %floor,fontsize=125)
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
#%% TEST 
# summarize model.
model.summary()
# load test data 
#%% test (Same hysteretic system)
test_kobe = pd.read_excel('KOBE.xlsx')
test_kobe = test_kobe.drop([0,1])
test_kobe = test_kobe.iloc[:, 1:10]
test_kobe = test_kobe.reset_index(drop=True)
test_kobe = test_kobe.astype('float')
test_kobe = test_kobe.to_numpy(dtype=float)
test_kobe = test_kobe/1000
test_kobe = test_kobe[2000:6000]

test_newzealand = pd.read_excel('NEW ZEALAND.xlsx')
test_newzealand = test_newzealand.drop([0,1])
test_newzealand = test_newzealand.iloc[:, 1:10]
test_newzealand = test_newzealand.reset_index(drop=True)
test_newzealand = test_newzealand.astype('float')
test_newzealand = test_newzealand.to_numpy(dtype=float)
test_newzealand = test_newzealand/1000
test_newzealand = test_newzealand[3000:7500]

test_hualien = pd.read_excel('2018 HUALIEN.xlsx')
test_hualien = test_hualien.drop([0,1])
test_hualien = test_hualien.iloc[:, 1:10]
test_hualien = test_hualien.reset_index(drop=True)
test_hualien = test_hualien.astype('float')
test_hualien = test_hualien.to_numpy(dtype=float)
test_hualien = test_hualien/1000
test_hualien = test_hualien[8250:13750]

test_chichi = pd.read_excel('CHICHI.xlsx')
test_chichi = test_chichi.drop([0,1])
test_chichi = test_chichi.iloc[:, 1:10]
test_chichi = test_chichi.reset_index(drop=True)
test_chichi = test_chichi.astype('float')
test_chichi = test_chichi.to_numpy(dtype=float)
test_chichi = test_chichi/1000
test_chichi = test_chichi[4000:19500]
#%% test (Different hysteretic system)
test_kobe = pd.read_excel('../model2/KOBE.xlsx')
test_kobe = test_kobe.drop([0,1])
test_kobe = test_kobe.iloc[:, 1:10]
test_kobe = test_kobe.reset_index(drop=True)
test_kobe = test_kobe.astype('float')
test_kobe = test_kobe.to_numpy(dtype=float)
test_kobe = test_kobe/1000
test_kobe = test_kobe[2000:6000]

test_newzealand = pd.read_excel('../model2/NEW ZEALAND.xlsx')
test_newzealand = test_newzealand.drop([0,1])
test_newzealand = test_newzealand.iloc[:, 1:10]
test_newzealand = test_newzealand.reset_index(drop=True)
test_newzealand = test_newzealand.astype('float')
test_newzealand = test_newzealand.to_numpy(dtype=float)
test_newzealand = test_newzealand/1000
test_newzealand = test_newzealand[3000:7500]

test_hualien = pd.read_excel('../model2/2018 HUALIEN.xlsx')
test_hualien = test_hualien.drop([0,1])
test_hualien = test_hualien.iloc[:, 1:10]
test_hualien = test_hualien.reset_index(drop=True)
test_hualien = test_hualien.astype('float')
test_hualien = test_hualien.to_numpy(dtype=float)
test_hualien = test_hualien/1000
test_hualien = test_hualien[8250:13750]

test_chichi = pd.read_excel('../model2/CHICHI.xlsx')
test_chichi = test_chichi.drop([0,1])
test_chichi = test_chichi.iloc[:, 1:10]
test_chichi = test_chichi.reset_index(drop=True)
test_chichi = test_chichi.astype('float')
test_chichi = test_chichi.to_numpy(dtype=float)
test_chichi = test_chichi/1000
test_chichi = test_chichi[4000:19500]

#%%  test datapreprocessing
#test1=test_kobe
test1=test_newzealand
#test1 = test_hualien
#test1=test_chichi
plt.plot(test1)    
plt.show()
# test1 = test1 * 9.81  #convert to m/s2 
print(test1.shape)

#%% absolute acc
for i in range(test1.shape[1]-1):
    test1[:,i]=test1[:,i]+test1[:,8]
#%%
# select input, 1st, and 8th floor to predict 8th floor
# data preprocessing (units:g)
# scaler = MinMaxScaler(feature_range=(-1,1))
test_data = test1.T
test_input = test_data[8,:]
first_test = test_data[7,:]
second_test = test_data[6,:]
third_test = test_data[5,:]
fourth_test = test_data[4,:]
fifth_test = test_data[3,:]
sixth_test = test_data[2,:]
seventh_test = test_data[1,:]
eighth_test= test_data[0,:]

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
test = np.hstack((first_test, fourth_test, eighth_test))
# split sequences by time steps
n_steps = 10
testX1, testy = split_multisequence(test, n_steps-1)
testX2 = split_sequence(test_input, n_steps)
print(testX1.shape, testX2.shape, testy.shape)
# padding the sequences
testXpad = np.full((testX1.shape[0], n_steps, testX1.shape[2]), fill_value=special_value)
for s, x in enumerate(testX1):
    seq_len = x.shape[0]
    testXpad[s, 0:seq_len, :] = x
# concatenate along the third axis
testX = np.dstack((testX2, testXpad))

# reshape from [samples, timesteps] into [samples, timesteps, features]
testX = testX.reshape((testX.shape[0], testX.shape[1], n_features))
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)
print('testX:',testX.shape)
print('testy:',testy.shape)

#%% Results
# predict test data
# test
yhat = model.predict(testX, verbose=0)
print(yhat)
print(yhat.shape)
print('test MSE:',mean_squared_error(testy, yhat))
print('test R-square:',r2_score(testy, yhat))
# demonstrate test prediction
for i in range(yhat.shape[1]):
    # test
    floor=fl[i]
    pred_fl = yhat[:,i]
    target = testy[:,i]
    t1 = np.linspace(0, len(target), len(target), endpoint=False)
    t2 = np.linspace(0, len(target), len(pred_fl), endpoint=False)
    fig = plt.figure(figsize=(90,36))
    ax1 = plt.subplot(211)
    ax1.plot(t1, target, label='observed', linestyle='-', color = 'k', linewidth=7.5)
    ax1.plot(t2, pred_fl, label='forecast', linestyle=':', color = 'b', linewidth=7.5)
    ax1.set_xlim(0,len(target))
    ax1.set_ylabel("Absolute acceleration (m/$s^2$)",fontsize=80)
    ax1.set_title("%d floor" %floor,fontsize=125)
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
    testy_fl = testy[:,i]
    for j in range(len(pred_fl)):
        abs_error[j] = np.abs((pred_fl[j] - testy_fl[j]))
    ax2.plot(abs_error, linewidth=7.5)
    #ax2.set_ylim(0,0.015)
    ax2.set_ylim(0,abs_error.max())
    ax2.set_xlabel("numbers of time points", fontsize=80)
    ax2.set_ylabel("Absolute error", fontsize=80)
    ax2.spines['left'].set_linewidth(5)
    ax2.spines['right'].set_linewidth(5)
    ax2.spines['bottom'].set_linewidth(5)
    ax2.spines['top'].set_linewidth(5)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.show()
    print('test MSE for %d floor:'%(i+1),mean_squared_error(testy_fl, pred_fl))
#%% save train and test error
train_error=train_predict-trainy
test_error=yhat-testy
np.save('train_error_model1(lstm)',train_error)
np.save('test_error_model1(lstm)',test_error)
# np.save(test_error_model2(lstm),test_error) #different hysteretic

#%% save model
model.save('lstm_nonlineaer_model1.h5')
print('save to the disk successfully!!')