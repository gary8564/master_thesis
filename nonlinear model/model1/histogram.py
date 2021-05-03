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
train_err_lstm=np.load('train_error_incomplete_steel_hysteretic(lstm).npy')
test_err_lstm=np.load('test_error_incomplete_newzealand_steel_hysteretic(lstm).npy')
train_err_mlp=np.load('train_error_incomplete_steel_hysteretic(mlp).npy')
test_err_mlp=np.load('test_error_incomplete_steel_hysteretic_newzealand(mlp).npy')
#test_err_mlp=-test_err_mlp
#test_err_mlp=np.load('test_error_incomplete_hualien_steel_hysteretic(mlp).npy')
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
    plt.title('Probability Density Funcion of training error at %d floor'%floor)
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
    hist=sns.distplot(test_err_fl_mlp, bins=100, hist=True, label='mlp',color='C1')
    hist=sns.distplot(test_err_fl_lstm, bins=100, hist=True, label='lstm',color='C0')
    plt.xlabel('Error')
    plt.ylabel('Density')
    floor=fl[i]
    plt.title('Probability Density Funcion of test error at %d floor'%floor)
    plt.legend()
    plt.show()
#%% histogram & dipersion index
test_err_lstm1=np.load('test_error_incomplete_newzealand(lstm).npy')
test_err_lstm1=-test_err_lstm1
test_err_mlp1=np.load('test_error_incomplete_newzealand(mlp).npy')
#test_err_mlp1=-test_err_mlp1
'''
train_err_lstm=np.abs(train_err_lstm)
train_err_mlp=np.abs(train_err_mlp)
test_err_lstm=np.abs(test_err_lstm)
test_err_mlp=np.abs(test_err_mlp)
test_err_lstm1=np.abs(test_err_lstm1)
test_err_mlp1=np.abs(test_err_mlp1)
'''
#%% Determine truncating number of sample
T=0.002 # time interval (sampling rate)
Ts_chichi=1/1.862
Ns_chichi=int(np.ceil(2.5*Ts_chichi/T))
Ts_norica=1/1.953
Ns_norica=int(np.ceil(2.5*Ts_norica/T))
Ts_jp311=1/1.266
Ns_jp311=int(np.ceil(2.5*Ts_jp311/T))
Ts_turkey=1/1.251
Ns_turkey=int(np.ceil(2.5*Ts_turkey/T))
Ts_amberley=1/0.6714
Ns_amberley=int(np.ceil(2.5*Ts_amberley/T))
Ts_california20=1/3.54
Ns_california20=int(np.ceil(2.5*Ts_california20/T))
Ts_331hualien=1/1.129
Ns_331hualien=int(np.ceil(2.5*Ts_331hualien/T))
Ts_chiayi=1/0.946
Ns_chiayi=int(np.ceil(2.5*Ts_chiayi/T))
Ts_yilan=1/2.38
Ns_yilan=int(np.ceil(2.5*Ts_yilan/T))
Ts_rome=1/1.801
Ns_rome=int(np.ceil(2.5*Ts_rome/T))
Ts_sakaria=1/0.9766
Ns_sakaria=int(np.ceil(2.5*Ts_sakaria/T))
Ts_athens=1/1.282
Ns_athens=int(np.ceil(2.5*Ts_athens/T))
Ts_kocaeli=1/0.4272
Ns_kocaeli=int(np.ceil(2.5*Ts_kocaeli/T))
Ts_meinung=1/1.16
Ns_meinung=int(np.ceil(2.5*Ts_meinung/T))
Ts_northridge=1/0.9766
Ns_northridge=int(np.ceil(2.5*Ts_northridge/T))
Ts_emeryville=1/1.465
Ns_emeryville=int(np.ceil(2.5*Ts_emeryville/T))
Ts_hollister=1/3.906
Ns_hollister=int(np.ceil(2.5*Ts_hollister/T))
Ts_corralitos=1/2.93
Ns_corralitos=int(np.ceil(2.5*Ts_corralitos/T))
Ts_friulli=1/0.7324
Ns_friulli=int(np.ceil(2.5*Ts_friulli/T))
Ts_elcentro=1/1.709
Ns_elcentro=int(np.ceil(2.5*Ts_elcentro/T))

Ns=np.mean([Ns_chichi,Ns_norica,Ns_corralitos,Ns_hollister,Ns_emeryville,
           Ns_jp311,Ns_northridge,Ns_friulli,Ns_california20,Ns_amberley,
           Ns_331hualien,Ns_chiayi,Ns_yilan,Ns_rome,Ns_sakaria,Ns_athens,Ns_kocaeli,
           Ns_meinung,Ns_turkey,Ns_elcentro])

Ns=int(np.ceil(Ns))
# %% LSTM
# calculate shannon entropy index
# train error
train_error_chichi = train_err_lstm[2000:2000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_chiayi = train_err_lstm[10350:10350+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_hualien331 = train_err_lstm[14000:14000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_meinung = train_err_lstm[22000:22000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_california20 = train_err_lstm[28500:28500+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_yilan=train_err_lstm[34000:34000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_amberley = train_err_lstm[38500:38500+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_athens = train_err_lstm[67000:67000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_corralitos = train_err_lstm[72000:72000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_elcentro = train_err_lstm[79000:79000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_emeryville = train_err_lstm[87000:87000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_friulli = train_err_lstm[97000:97000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_hollister = train_err_lstm[101000:101000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_jp311 = train_err_lstm[110000:110000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_kocaeli = train_err_lstm[127000:127000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_2011turkey = train_err_lstm[135000:135000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_northridge = train_err_lstm[140000:140000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_rome = train_err_lstm[147000:147000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_sakaria = train_err_lstm[147000:147000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error_norica = train_err_lstm[151000:151000+Ns].reshape(1,-1,train_err_lstm.shape[1])
train_error = np.concatenate((train_error_chichi,train_error_chiayi,train_error_hualien331,train_error_meinung,
                              train_error_california20,train_error_yilan,train_error_amberley,
                     train_error_athens,train_error_corralitos,train_error_elcentro,train_error_emeryville,train_error_friulli,
                     train_error_hollister,train_error_jp311,train_error_kocaeli,train_error_2011turkey,train_error_northridge,
                     train_error_rome,train_error_sakaria,train_error_norica),axis=0)
train_error = np.mean(train_error,axis=0)

# test difference
# health
test_error_h=test_err_lstm[500:500+Ns,:]
# damage
test_error=test_err_lstm1
test_error=test_error[500:500+Ns,:]
# 1st floor
train_error_1=train_error[:,0]
error_1_h=test_error_h[:,0]
error_1=test_error[:,0]
plt.figure()
(n_1, bins_1, patches_1) = plt.hist(error_1, bins=100)
plt.xlabel('Absolute error')
plt.ylabel('Counts')
plt.title('Frequeny Histogram of test error in first floor')
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
#damage_idx_train1=np.var(x1)/np.mean(x1)

hist=sns.distplot(error_1_h, bins=100, hist=False, label='bouc wen model 1')

x,y = hist.get_lines()[1].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='darkorange')
#damage_idx_h1=np.var(x)/np.mean(x)

hist=sns.distplot(error_1, bins=100, hist=False, label='bouc wen model 2')

x,y = hist.get_lines()[2].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='forestgreen')
#damage_idx_d1=np.var(x)/np.mean(x)
hist.set(xlabel='Error')
plt.title('1st floor', fontsize=25)
plt.legend()
plt.show()


# 4th floor
train_error_3=train_error[:,1]
error_3_h=test_error_h[:,1]
error_3=test_error[:,1]
plt.figure()
(n_3, bins_3, patches_3) = plt.hist(error_3, bins=100)
plt.xlabel('Absolute error')
plt.ylabel('Counts')
plt.title('Frequeny Histogram of test error in fourth floor')
plt.show()
plt.figure()
sns.set_style('darkgrid')
hist=sns.distplot(train_error_3, bins=100, hist=False, label='train')

x1,y1 = hist.get_lines()[0].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x1[nearest_05]
y_3sigma = y1[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='royalblue')
#damage_idx_train3=np.var(x1)/np.mean(x1)

hist=sns.distplot(error_3_h, bins=100, hist=False, label='bouc wen model 1')

x,y = hist.get_lines()[1].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='darkorange')
#damage_idx_h3=np.var(x)/np.mean(x)

hist=sns.distplot(error_3, bins=100,hist=False,label='bouc wen model 2')

x,y = hist.get_lines()[2].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='forestgreen')
#damage_idx_d3=np.var(x)/np.mean(x)
hist.set(xlabel='Error')
plt.title('4th floor', fontsize=25)
plt.legend()
plt.show()

# 8th floor
train_error_8=train_error[:,2]
error_8_h=test_error_h[:,2]
error_8=test_error[:,2]
plt.figure()
(n_8, bins_8, patches_8) = plt.hist(error_8, bins=100)
plt.xlabel('Absolute error')
plt.ylabel('Counts')
plt.title('Frequeny Histogram of test error in eighth floor')
plt.show()
plt.figure()
sns.set_style('darkgrid')
hist=sns.distplot(train_error_8, bins=100, hist=False, label='train')

x1,y1 = hist.get_lines()[0].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x1[nearest_05]
y_3sigma = y1[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='royalblue')
#damage_idx_train8=np.var(x1)/np.mean(x1)

hist=sns.distplot(error_8_h, bins=100,hist=False,label='bouc wen model 1')

x,y = hist.get_lines()[1].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='darkorange')
#damage_idx_h8=np.var(x)/np.mean(x)

hist=sns.distplot(error_8, bins=100,hist=False,label='bou wen model 2')

x,y = hist.get_lines()[2].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='forestgreen')
#damage_idx_d8=np.var(x)/np.mean(x)
hist.set(xlabel='Error')
plt.title('8th floor', fontsize=25)
plt.legend()
plt.show()

# boxplot
# 1st floor
error_1_h=test_error_h[:,0]
error_1=test_error[:,0]
d1 = {'train':train_error_1,'health':error_1_h,'damage':error_1}
df1 = pd.DataFrame(data=d1)
#df1.boxplot()
sns.boxplot(x="variable", y="value", data=pd.melt(df1))
plt.show()
median1_train = np.median(train_error_1)
upper_quartile1_train = np.percentile(train_error_1, 75)
lower_quartile1_train = np.percentile(train_error_1, 25)
print('train 1st floor:',upper_quartile1_train-lower_quartile1_train)
median1_h = np.median(error_1_h)
upper_quartile1_h = np.percentile(error_1_h, 75)
lower_quartile1_h = np.percentile(error_1_h, 25)
print('health 1st floor:',upper_quartile1_h-lower_quartile1_h)
median1 = np.median(error_1)
upper_quartile1 = np.percentile(error_1, 75)
lower_quartile1 = np.percentile(error_1, 25)
print('damage 1st floor:',upper_quartile1-lower_quartile1)

# 4th floor
error_4_h=test_error_h[:,1]
error_4=test_error[:,1]
d3 = {'train':train_error_3,'health':error_3_h,'damage':error_3}
df3 = pd.DataFrame(data=d3)
#df3.boxplot()
sns.boxplot(x="variable", y="value", data=pd.melt(df3))
plt.show()
median3_train = np.median(train_error_3)
upper_quartile3_train = np.percentile(train_error_3, 75)
lower_quartile3_train = np.percentile(train_error_3, 25)
print('train 4th floor:',upper_quartile3_train-lower_quartile3_train)
median3_h = np.median(error_3_h)
upper_quartile3_h = np.percentile(error_3_h, 75)
lower_quartile3_h = np.percentile(error_3_h, 25)
print('health 4th floor:',upper_quartile3_h-lower_quartile3_h)
median3 = np.median(error_3)
upper_quartile3 = np.percentile(error_3, 75)
lower_quartile3 = np.percentile(error_3, 25)
print('damage 4th floor:',upper_quartile3-lower_quartile3)


# 8th floor
error_8_h=test_error_h[:,2]
error_8=test_error[:,2]
d8 = {'train':train_error_8,'health':error_8_h,'damage':error_8}
df8 = pd.DataFrame(data=d8)
#df8.boxplot()
sns.boxplot(x="variable", y="value", data=pd.melt(df8))
plt.show()
median8_train = np.median(train_error_8)
upper_quartile8_train = np.percentile(train_error_8, 75)
lower_quartile8_train = np.percentile(train_error_8, 25)
print('train 8th floor:',upper_quartile8_train-lower_quartile8_train)
median8_h = np.median(error_8_h)
upper_quartile8_h = np.percentile(error_8_h, 75)
lower_quartile8_h = np.percentile(error_8_h, 25)
print('health 8th floor:',upper_quartile8_h-lower_quartile8_h)
median8 = np.median(error_8)
upper_quartile8 = np.percentile(error_8, 75)
lower_quartile8 = np.percentile(error_8, 25)
print('damage 8th floor:',upper_quartile8-lower_quartile8)
# %% MLP
# calculate shannon entropy index
# train error
train_error_chichi = train_err_mlp[2000:2000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_chiayi = train_err_mlp[10350:10350+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_hualien331 = train_err_mlp[14000:14000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_meinung = train_err_mlp[22000:22000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_california20 = train_err_mlp[28500:28500+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_yilan=train_err_mlp[34000:34000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_amberley = train_err_mlp[38500:38500+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_athens = train_err_mlp[67000:67000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_corralitos = train_err_mlp[72000:72000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_elcentro = train_err_mlp[79000:79000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_emeryville = train_err_mlp[87000:87000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_friulli = train_err_mlp[97000:97000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_hollister = train_err_mlp[101000:101000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_jp311 = train_err_mlp[110000:110000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_kocaeli = train_err_mlp[127000:127000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_2011turkey = train_err_mlp[135000:135000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_northridge = train_err_mlp[140000:140000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_rome = train_err_mlp[147000:147000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_sakaria = train_err_mlp[147000:147000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error_norica = train_err_mlp[151000:151000+Ns].reshape(1,-1,train_err_mlp.shape[1])
train_error = np.concatenate((train_error_chichi,train_error_chiayi,train_error_hualien331,train_error_meinung,
                              train_error_california20,train_error_yilan,train_error_amberley,
                     train_error_athens,train_error_corralitos,train_error_elcentro,train_error_emeryville,train_error_friulli,
                     train_error_hollister,train_error_jp311,train_error_kocaeli,train_error_2011turkey,train_error_northridge,
                     train_error_rome,train_error_sakaria,train_error_norica),axis=0)
train_error = np.mean(train_error,axis=0)

# test difference
# health
test_error_h=test_err_mlp[500:500+Ns,:]
# damage
test_error=test_err_mlp1
test_error=test_error[500:500+Ns,:]
# 1st floor
train_error_1=train_error[:,0]
error_1_h=test_error_h[:,0]
error_1=test_error[:,0]
plt.figure()
(n_1, bins_1, patches_1) = plt.hist(error_1, bins=100)
plt.xlabel('Error')
plt.ylabel('Counts')
plt.title('Frequeny Histogram of test error in first floor')
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
#damage_idx_train1=np.var(x1)/np.mean(x1)

hist=sns.distplot(error_1_h, bins=100, hist=False, label='bouc wen model 1')

x,y = hist.get_lines()[1].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='darkorange')
damage_idx_h1=np.var(x)/np.mean(x)

hist=sns.distplot(error_1, bins=100, hist=False, label='bouc wen model 2')

x,y = hist.get_lines()[2].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='forestgreen')
damage_idx_d1=np.var(x)/np.mean(x)
hist.set(xlabel='Error')
plt.title('1st floor', fontsize=25)
plt.legend()
plt.show()


# 4th floor
train_error_3=train_error[:,1]
error_3_h=test_error_h[:,1]
error_3=test_error[:,1]
plt.figure()
(n_3, bins_3, patches_3) = plt.hist(error_3, bins=100)
plt.xlabel('Absolute error')
plt.ylabel('Counts')
plt.title('Frequeny Histogram of test error in fourth floor')
plt.show()
plt.figure()
sns.set_style('darkgrid')
hist=sns.distplot(train_error_3, bins=100, hist=False, label='train')

x1,y1 = hist.get_lines()[0].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x1[nearest_05]
y_3sigma = y1[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='royalblue')
damage_idx_train3=np.var(x1)/np.mean(x1)

hist=sns.distplot(error_3_h, bins=100, hist=False, label='bouc wen model 1')

x,y = hist.get_lines()[1].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='darkorange')
damage_idx_h3=np.var(x)/np.mean(x)

hist=sns.distplot(error_3, bins=100,hist=False,label='bouc wen model 2')

x,y = hist.get_lines()[2].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='forestgreen')
damage_idx_d3=np.var(x)/np.mean(x)
hist.set(xlabel='Error')
plt.title('4th floor', fontsize=25)
plt.legend()
plt.show()

# 8th floor
train_error_8=train_error[:,2]
error_8_h=test_error_h[:,2]
error_8=test_error[:,2]
plt.figure()
(n_8, bins_8, patches_8) = plt.hist(error_8, bins=100)
plt.xlabel('Error')
plt.ylabel('Counts')
plt.title('Frequeny Histogram of test error in eighth floor')
plt.show()
plt.figure()
sns.set_style('darkgrid')
hist=sns.distplot(train_error_8, bins=100, hist=False, label='train')

x1,y1 = hist.get_lines()[0].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y1, x1, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x1[nearest_05]
y_3sigma = y1[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='royalblue')
damage_idx_train8=np.var(x1)/np.mean(x1)

hist=sns.distplot(error_8_h, bins=100,hist=False,label='bouc wen model 1')

x,y = hist.get_lines()[1].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='darkorange')
damage_idx_h8=np.var(x)/np.mean(x)

hist=sns.distplot(error_8, bins=100,hist=False,label='bouc wen model 2')

x,y = hist.get_lines()[2].get_data()
#care with the order, it is first y
#initial fills a 0 so the result has same length than x
cdf = scipy.integrate.cumtrapz(y, x, initial=0)
nearest_05 = np.abs(cdf-0.75).argmin()
x_3sigma = x[nearest_05]
y_3sigma = y[nearest_05]
plt.vlines(x_3sigma, 0, np.max(y1),linestyles='--',color='forestgreen')
damage_idx_d8=np.var(x)/np.mean(x)
hist.set(xlabel='Error')
plt.title('8th floor', fontsize=25)
plt.legend()
plt.show()

# boxplot
# 1st floor
error_1_h=test_error_h[:,0]
error_1=test_error[:,0]
d1 = {'train':train_error_1,'health':error_1_h,'damage':error_1}
df1 = pd.DataFrame(data=d1)
#df1.boxplot()
sns.boxplot(x="variable", y="value", data=pd.melt(df1))
plt.show()
median1_train = np.median(train_error_1)
upper_quartile1_train = np.percentile(train_error_1, 75)
lower_quartile1_train = np.percentile(train_error_1, 25)
print('train 1st floor:',upper_quartile1_train-lower_quartile1_train)
median1_h = np.median(error_1_h)
upper_quartile1_h = np.percentile(error_1_h, 75)
lower_quartile1_h = np.percentile(error_1_h, 25)
print('health 1st floor:',upper_quartile1_h-lower_quartile1_h)
median1 = np.median(error_1)
upper_quartile1 = np.percentile(error_1, 75)
lower_quartile1 = np.percentile(error_1, 25)
print('damage 1st floor:',upper_quartile1-lower_quartile1)

# 4th floor
error_4_h=test_error_h[:,1]
error_4=test_error[:,1]
d3 = {'train':train_error_3,'health':error_3_h,'damage':error_3}
df3 = pd.DataFrame(data=d3)
#df3.boxplot()
sns.boxplot(x="variable", y="value", data=pd.melt(df3))
plt.show()
median3_train = np.median(train_error_3)
upper_quartile3_train = np.percentile(train_error_3, 75)
lower_quartile3_train = np.percentile(train_error_3, 25)
print('train 4th floor:',upper_quartile3_train-lower_quartile3_train)
median3_h = np.median(error_3_h)
upper_quartile3_h = np.percentile(error_3_h, 75)
lower_quartile3_h = np.percentile(error_3_h, 25)
print('health 4th floor:',upper_quartile3_h-lower_quartile3_h)
median3 = np.median(error_3)
upper_quartile3 = np.percentile(error_3, 75)
lower_quartile3 = np.percentile(error_3, 25)
print('damage 4th floor:',upper_quartile3-lower_quartile3)


# 8th floor
error_8_h=test_error_h[:,2]
error_8=test_error[:,2]
d8 = {'train':train_error_8,'health':error_8_h,'damage':error_8}
df8 = pd.DataFrame(data=d8)
#df8.boxplot()
sns.boxplot(x="variable", y="value", data=pd.melt(df8))
plt.show()
median8_train = np.median(train_error_8)
upper_quartile8_train = np.percentile(train_error_8, 75)
lower_quartile8_train = np.percentile(train_error_8, 25)
print('train 8th floor:',upper_quartile8_train-lower_quartile8_train)
median8_h = np.median(error_8_h)
upper_quartile8_h = np.percentile(error_8_h, 75)
lower_quartile8_h = np.percentile(error_8_h, 25)
print('health 8th floor:',upper_quartile8_h-lower_quartile8_h)
median8 = np.median(error_8)
upper_quartile8 = np.percentile(error_8, 75)
lower_quartile8 = np.percentile(error_8, 25)
print('damage 8th floor:',upper_quartile8-lower_quartile8)