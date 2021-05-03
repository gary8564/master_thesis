# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:10:35 2020

@author: CSHuangLab
"""

# %% import modules
import numpy as np
import pandas as pd
import math
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from scipy.stats import entropy
import scipy
import seaborn as sns

import random
SEED=5278
random.seed(SEED)
np.random.seed(SEED)

#%% load error file
train_err_lstm=np.load('train_error(lstm).npy')
test_err_lstm=np.load('error(lstm).npy')
train_err_bilstm=np.load('train_error(bilstm).npy')
test_err_bilstm=np.load('error(bilstm).npy')
train_err_mlp=np.load('train_error(mlp).npy')
test_err_mlp=np.load('error(mlp).npy')
#%% train
for i in range(train_err_lstm.shape[1]):
    train_err_fl_lstm=train_err_lstm[:,i]
    #train_err_fl_mlp=train_err_mlp[:,i]
    train_err_fl_bilstm=train_err_bilstm[:,i]    
    '''
    weights = np.ones_like(train_error_fl) / (len(train_error_fl))
    plt.figure()
    plt.hist(train_error_fl, bins=100, weights=weights)
    plt.xlabel('Absolute error')
    plt.ylabel('Counts')
    plt.title('Frequeny Histogram of test error in first floor')
    plt.show()
    '''
    plt.figure()
    sns.set_style('darkgrid')
    #hist=sns.distplot(train_err_fl_mlp, bins=100, hist=True, label='mlp',color='C1')
    hist=sns.distplot(train_err_fl_lstm, bins=100, hist=True, label='lstm',color='C0')
    hist=sns.distplot(train_err_fl_bilstm, bins=100, hist=True, label='bi-lstm',color='C3')
    plt.xlabel('Error',fontsize=15)
    plt.ylabel('Density',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Probability Density Funcion of training error in %d floor'%(i+1))
    plt.title('Probability Density Funcion of training error', fontsize=15)
    #plt.legend(loc=1,prop={"size":15})
    plt.legend(prop={'size': 15})
    plt.show()
#%% test
for i in range(test_err_lstm.shape[1]):
    test_err_fl_lstm=test_err_lstm[:,i]
    #test_err_fl_mlp=test_err_mlp[:,i]  
    test_err_fl_bilstm=test_err_bilstm[:,i]  
    '''
    weights = np.ones_like(test_error_fl) / (len(test_error_fl))
    plt.figure()
    plt.hist(test_error_fl, bins=100, weights=weights)
    plt.xlabel('Absolute error')
    plt.ylabel('Counts')
    plt.title('Frequeny Histogram of test error in first floor')
    plt.show()
    '''
    plt.figure()
    sns.set_style('darkgrid')
    #hist=sns.distplot(test_err_fl_mlp, bins=100, hist=True, label='mlp',color='C1')
    hist=sns.distplot(test_err_fl_lstm, bins=100, hist=True, label='lstm',color='C0')
    hist=sns.distplot(test_err_fl_bilstm, bins=100, hist=True, label='bi-lstm',color='C3')
    plt.xlabel('Error',fontsize=15)
    plt.ylabel('Density',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Turkey', fontsize=18)
    #plt.title('Probability Density Funcion of test error in %d floor'%(i+1))
    plt.legend(prop={'size': 15})
    plt.show()