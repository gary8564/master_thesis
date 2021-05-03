# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:23:32 2020

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
test_err_lstm=np.load('test_error(lstm).npy')
train_err_mlp=np.load('train_error(mlp).npy')
test_err_mlp=np.load('test_error(mlp).npy')
#%% train
fl=np.array([1,4,8])
for i in range(train_err_lstm.shape[1]):
    train_err_fl_lstm=train_err_lstm[:,i]
    train_err_fl_mlp=train_err_mlp[:,i]
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
    hist=sns.distplot(train_err_fl_mlp, bins=100, hist=True, label='mlp',color='C1')
    hist=sns.distplot(train_err_fl_lstm, bins=100, hist=True, label='lstm',color='C0')
    plt.xlabel('Error')
    plt.ylabel('Density')
    floor=fl[i]
    plt.title('Probability Density Funcion of training error in %d floor'%(floor))
    plt.legend()
    plt.show()
#%% test
fl=np.array([1,4,8])
for i in range(test_err_lstm.shape[1]):
    test_err_fl_lstm=test_err_lstm[:,i]
    test_err_fl_mlp=test_err_mlp[:,i]    
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
    hist=sns.distplot(test_err_fl_mlp, bins=100, hist=True, label='600~700gal',color='C3')
    hist=sns.distplot(test_err_fl_lstm, bins=100, hist=True, label='raw',color='C0')
    plt.xlabel('Error')
    plt.ylabel('Density')
    floor=fl[i]
    plt.title('Probability Density Funcion of test error in %d floor'%floor)
    plt.legend()
    plt.show()