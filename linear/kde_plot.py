# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:10:35 2020

@author: CSHuangLab
"""

# %% import modules
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy
import random
import pandas as pd
SEED=5278
random.seed(SEED)
np.random.seed(SEED)

#%% Plot KDE in Thesis Chapter 3 (to compare lstm and mlp)
#load error file
train_err_lstm=np.load('train_error_lstm.npy')
test_err_lstm=np.load('test_error_health_lstm.npy')
#train_err_bilstm=np.load('train_error_bilstm.npy')
#test_err_bilstm=np.load('error_bilstm_kobe.npy')
train_err_mlp=np.load('train_error_mlp.npy')
test_err_mlp=np.load('test_error_health_mlp.npy')
#train plot
for i in range(train_err_lstm.shape[1]):
    train_err_fl_lstm=train_err_lstm[:,i]
    train_err_fl_mlp=train_err_mlp[:,i]
    #train_err_fl_bilstm=train_err_bilstm[:,i]    
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
    #hist=sns.distplot(train_err_fl_bilstm, bins=100, hist=True, label='bi-lstm',color='C3')
    plt.xlabel('Error',fontsize=15)
    plt.ylabel('Density',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Probability Density Funcion of training error', fontsize=15)
    #plt.legend(loc=1,prop={"size":15})
    plt.legend(prop={'size': 15})
    plt.show()
#test plot
for i in range(test_err_lstm.shape[1]):
    test_err_fl_lstm=test_err_lstm[:,i]
    test_err_fl_mlp=test_err_mlp[:,i]  
    #test_err_fl_bilstm=test_err_bilstm[:,i]  
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
    hist=sns.distplot(test_err_fl_mlp, bins=100, hist=True, label='mlp',color='C1')
    hist=sns.distplot(test_err_fl_lstm, bins=100, hist=True, label='lstm',color='C0')
    #hist=sns.distplot(test_err_fl_bilstm, bins=100, hist=True, label='bi-lstm',color='C3')
    plt.xlabel('Error',fontsize=15)
    plt.ylabel('Density',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Kobe', fontsize=18)
    plt.legend(prop={'size': 15})
    plt.show()
#%% Plot KDE in thesis Chpater 4
## load error data (lstm/mlp)
train_error=np.load('train_error_lstm.npy') # if mlp, change file name lstm to mlp
test_error_h=np.load('test_error_health_lstm.npy') 
test_error_d=np.load('test_error_damaged_lstm.npy') 
#%% Determine truncating number of sample
T=0.002 # time interval(sampling rate)
t_truncate=2 #unit:sec
Ns=t_truncate/T
#%% KDE plot train error
train_error_chichi = train_error[1350:1350+Ns].reshape(1,-1,train_error.shape[1])
train_error_norica = train_error[6260:6260+Ns].reshape(1,-1,train_error.shape[1])
train_error_mexico = train_error[10600:10600+Ns].reshape(1,-1,train_error.shape[1])
train_error_berkeley = train_error[15750:15750+Ns].reshape(1,-1,train_error.shape[1])
train_error_christchurch = train_error[19250:19250+Ns].reshape(1,-1,train_error.shape[1])
train_error_christchurch11 = train_error[24250:24250+Ns].reshape(1,-1,train_error.shape[1])
train_error_jp311 = train_error[31500:31500+Ns].reshape(1,-1,train_error.shape[1])
train_error_hualien19 = train_error[44250:44250+Ns].reshape(1,-1,train_error.shape[1])
train_error_amberley = train_error[52250:52250+Ns].reshape(1,-1,train_error.shape[1])
train_error_nantou = train_error[81000:81000+Ns].reshape(1,-1,train_error.shape[1])
train_error_hualien = train_error[85750:85750+Ns].reshape(1,-1,train_error.shape[1])
train_error_california20 = train_error[94500:94500+Ns].reshape(1,-1,train_error.shape[1])
train_error_331hualien = train_error[96700:96700+Ns].reshape(1,-1,train_error.shape[1])
train_error_chile = train_error[102700:102700+Ns].reshape(1,-1,train_error.shape[1])
train_error_chiayi = train_error[106950:106950+Ns].reshape(1,-1,train_error.shape[1])
train_error_yilan = train_error[110700:110700+Ns].reshape(1,-1,train_error.shape[1])
train_error_rome = train_error[113450:113450+Ns].reshape(1,-1,train_error.shape[1])
train_error_sakaria = train_error[115950:115950+Ns].reshape(1,-1,train_error.shape[1])
train_error_athens = train_error[118450:118450+Ns].reshape(1,-1,train_error.shape[1])
train_error_kocaeli = train_error[123751:123751+Ns].reshape(1,-1,train_error.shape[1])
train_error_meinung = train_error[127951:127951+Ns].reshape(1,-1,train_error.shape[1])
train_error_602nantou = train_error[138451:138451+Ns].reshape(1,-1,train_error.shape[1])
train_error_california19 = train_error[147951:147951+Ns].reshape(1,-1,train_error.shape[1])

train_error = np.vstack((train_error_chichi,train_error_norica,train_error_mexico,train_error_berkeley,
                         train_error_christchurch,train_error_christchurch11,train_error_jp311,
                         train_error_hualien19,train_error_amberley,train_error_nantou,
                         train_error_hualien,train_error_california20,train_error_331hualien,
                         train_error_chile,train_error_chiayi,train_error_yilan,train_error_rome,
                         train_error_sakaria,train_error_athens,train_error_kocaeli,train_error_meinung,
                         train_error_602nantou,train_error_california19))
'''
# for 15 cases
train_error_christchurch = train_error[19500:19500+Ns].reshape(1,-1,train_error.shape[1])
train_error_jp311 = train_error[28750:28750+Ns].reshape(1,-1,train_error.shape[1])
train_error_hualien19 = train_error[41750:41750+Ns].reshape(1,-1,train_error.shape[1])
train_error_amberley = train_error[49750:49750+Ns].reshape(1,-1,train_error.shape[1])
train_error_nantou = train_error[78250:78250+Ns].reshape(1,-1,train_error.shape[1])
train_error_hualien = train_error[83750:83750+Ns].reshape(1,-1,train_error.shape[1])
train_error_california20 = train_error[92000:92000+Ns].reshape(1,-1,train_error.shape[1])
train_error_331hualien = train_error[94200:94200+Ns].reshape(1,-1,train_error.shape[1])
train_error_chile = train_error[99750:99750+Ns].reshape(1,-1,train_error.shape[1])
train_error_chiayi = train_error[104450:104450+Ns].reshape(1,-1,train_error.shape[1])
train_error_yilan = train_error[108200:108200+Ns].reshape(1,-1,train_error.shape[1])
train_error = np.vstack((train_error_chichi,train_error_norica,train_error_mexico,train_error_berkeley,
                         train_error_christchurch,train_error_jp311,
                         train_error_hualien19,train_error_amberley,train_error_nantou,
                         train_error_hualien,train_error_california20,train_error_331hualien,
                         train_error_chile,train_error_chiayi,train_error_yilan))

# for 10 cases
train_error_chichi = train_error[1350:1350+Ns].reshape(1,-1,train_error.shape[1])
train_error_norica = train_error[6260:6260+Ns].reshape(1,-1,train_error.shape[1])
train_error_mexico = train_error[10600:10600+Ns].reshape(1,-1,train_error.shape[1])
train_error_berkeley = train_error[16000:16000+Ns].reshape(1,-1,train_error.shape[1])
train_error_christchurch11 = train_error[18250:18250+Ns].reshape(1,-1,train_error.shape[1])
train_error_jp311 = train_error[26500:26500+Ns].reshape(1,-1,train_error.shape[1])
train_error_hualien19= train_error[38250:38250+Ns].reshape(1,-1,train_error.shape[1])
train_error_nantou = train_error[41750:41750+Ns].reshape(1,-1,train_error.shape[1])
train_error_hualien = train_error[47600:47600+Ns].reshape(1,-1,train_error.shape[1])
train_error_california20 = train_error[55500:55500+Ns].reshape(1,-1,train_error.shape[1])
train_error = np.vstack((train_error_chichi,train_error_norica,train_error_mexico,train_error_berkeley,
                         train_error_christchurch11,train_error_jp311,
                         train_error_hualien19,train_error_nantou,
                         train_error_hualien,train_error_california20))
'''
train_error = np.mean(train_error,axis=0)
#%% KDE plot test health
lr_start=1000 # large test seismic response starting sample point
test_error_h=test_error_h[lr_start:lr_start+Ns,:]
# damage
test_error_d=test_error_d[lr_start:lr_start+Ns,:]
# pdf at each floor
for i in range(train_error.shape[1]):
    train_error_1=train_error[:,i]
    error_1_h=test_error_h[:,i]
    error_1=test_error_d[:,i]
    plt.figure()
    (n_1, bins_1, patches_1) = plt.hist(error_1, bins=100)
    plt.xlabel('Absolute error')
    plt.ylabel('Counts')
    plt.title('Frequeny Histogram of test error in %d floor'%(i+1))
    plt.show()
    plt.figure()
    sns.set_style('darkgrid')
    hist=sns.distplot(train_error_1, bins=100, hist=False, label='train')
    x1,y1 = hist.get_lines()[0].get_data()
    #care with the order, it is first y
    #initial fills a 0 so the result has same length than x
    cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
    nearest_05 = np.abs(cdf-0.75).argmin()
    x_3sigma = x1[nearest_05]
    y_3sigma = y1[nearest_05]
    plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='royalblue')
    di_train1=x_3sigma/np.amax(x1)
    hist=sns.distplot(error_1_h, bins=100, hist=False, label='health')
    x,y = hist.get_lines()[1].get_data()
    #care with the order, it is first y
    #initial fills a 0 so the result has same length than x
    cdf = scipy.integrate.cumtrapz(y, x, initial=0)
    nearest_05 = np.abs(cdf-0.75).argmin()
    x_3sigma = x[nearest_05]
    y_3sigma = y[nearest_05]
    plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='darkorange')
    di_h1=x_3sigma/np.amax(x)
    hist=sns.distplot(error_1, bins=100, hist=False, label='damage')
    x,y = hist.get_lines()[2].get_data()
    #care with the order, it is first y
    #initial fills a 0 so the result has same length than x
    cdf = scipy.integrate.cumtrapz(y, x, initial=0)
    nearest_05 = np.abs(cdf-0.75).argmin()
    x_3sigma = x[nearest_05]
    y_3sigma = y[nearest_05]
    plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='forestgreen')
    di_d1=x_3sigma/np.amax(x)
    hist.set(xlabel='Absolute error')
    plt.title('%d floor'%(i+1), fontsize=25)
    plt.legend()
    plt.show()
    
#%% boxplot
# test difference
# health
test_error_h=test_error_h[lr_start:lr_start+Ns,:]
# damage
test_error_d=test_error_d[lr_start:lr_start+Ns,:]
# each floor
for i in range(train_error.shape[1]):
    train_error_i=train_error[:,i]
    error_i_h=test_error_h[:,i]
    error_i=test_error_d[:,i]
    d1 = {'train':train_error_i,'health':error_i_h,'damage':error_i}
    df1 = pd.DataFrame(data=d1)
    #df1.boxplot()
    sns.boxplot(x="variable", y="value", data=pd.melt(df1))
    plt.title('%d floor'%(i+1),fontsize=25)
    plt.show()
    #train
    median_train = np.median(train_error_i)
    upper_quartile_train = np.percentile(train_error_i, 75)
    lower_quartile_train = np.percentile(train_error_i, 25)
    iqr_train=upper_quartile_train-lower_quartile_train
    
    #test health
    median_h = np.median(error_i_h)
    upper_quartile_h = np.percentile(error_i_h, 75)
    lower_quartile_h = np.percentile(error_i_h, 25)
    iqr_h=upper_quartile_h-lower_quartile_h
    iqr_ratio_h=iqr_h/iqr_train
    #test damage
    median = np.median(error_i)
    upper_quartile = np.percentile(error_i, 75)
    lower_quartile = np.percentile(error_i, 25)
    iqr=upper_quartile-lower_quartile
    iqr_ratio=iqr/iqr_train
    print('train IQR at %d floor:'%(i+1),iqr_train)
    print('health IQR %d floor:'%(i+1),iqr_h)
    print('damage IQR %d floor:'%(i+1),iqr)
    print('health D.I.(IQR) %d floor:'%(i+1),iqr_ratio_h)
    print('damage D.I.(IQR) at %d floor:'%(i+1),iqr_ratio)




