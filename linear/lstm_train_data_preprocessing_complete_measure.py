# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:21:06 2020

@author: CSHuangLab
"""
# %% Import modules
import numpy as np
from matplotlib import pyplot as plt
import os
#Force Tensorflow to use a single thread
import random
SEED=1996
random.seed(SEED)
np.random.seed(SEED)
# %% Define function    
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
        
        ## Split Ver 1
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
		# gather input and output parts of the pattern
        seq_x = sequence[i:end_ix]
        '''
        ## Split Ver 2
        # find the end of this pattern
        end_ix = i + n_steps + 1
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
		# gather input and output parts of the pattern
        seq_x = sequence[i+1:end_ix]
        '''
        X.append(seq_x)
    return np.array(X)
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


# %% Load data
path='./no noise'
# train
train_jiji = np.loadtxt(os.path.join(path,'00%_chichi_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_jiji)     # from the plot, we can guess the acceleration units should be g
plt.show()
train_jiji=train_jiji[1500:7500,:]

train_norica = np.loadtxt(os.path.join(path,'00%_norica_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_norica)    
plt.show()
train_norica=train_norica[2250:4500,:]

train_mexico = np.loadtxt(os.path.join(path,'00%_mexico_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_mexico)    
plt.show()
train_mexico=train_mexico[6000:13500,:]

train_berkeley = np.loadtxt(os.path.join(path,'00%_berkeley_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_berkeley)
plt.show()
train_berkeley=train_berkeley[2900:4900]

train_christchurch = np.loadtxt(os.path.join(path,'00%_11newzealand_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_christchurch)
plt.show()
train_christchurch=train_christchurch[1500:7500]

train_jp311 = np.loadtxt(os.path.join(path,'00%_japan311_1_8DOF_500Hz_a.dat'))
plt.plot(train_jp311)
plt.show()
train_jp311=train_jp311[5000:22500]

train_hualien19 = np.loadtxt(os.path.join(path,'00%_19hualien_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_hualien19)
plt.show()
train_hualien19=train_hualien19[4500:8000]

train_327nantou = np.loadtxt(os.path.join(path,'00%_327nantou_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_327nantou)
plt.show()
train_327nantou=train_327nantou[5500:10500]

train_amberley = np.loadtxt(os.path.join(path,'00%_amberley_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_amberley)
plt.show()
train_amberley=train_amberley[7500:40500]

train_1031hualien = np.loadtxt(os.path.join(path,'00%_131031hualien_responses_1_8DOF_500Hz_a.dat'))
train_1031hualien=train_1031hualien[3750:12500]
plt.plot(train_1031hualien)
plt.show()

train_california20 = np.loadtxt(os.path.join(path,'00%_20california_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_california20)
plt.show()
train_california20=train_california20[2800:5500]

train_331hualien = np.loadtxt(os.path.join(path,'00%_331TPE_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_331hualien)
plt.show()
train_331hualien=train_331hualien[3500:9000]

train_chile = np.loadtxt(os.path.join(path,'00%_chile_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_chile)
plt.show()
train_chile=train_chile[6250:11000]

train_chiayi = np.loadtxt(os.path.join(path,'00%_chiayi_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_chiayi)
plt.show()
train_chiayi=train_chiayi[5500:9000]

train_yilan = np.loadtxt(os.path.join(path,'00%_yilan_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_yilan)
plt.show()
train_yilan=train_yilan[3500:6000]

train_rome = np.loadtxt(os.path.join(path,'00%_Rome_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_rome)
plt.show()
train_rome=train_rome[1000:4000]

train_sakaria = np.loadtxt(os.path.join(path,'00%_Sakaria_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_sakaria)
plt.show()

train_athens = np.loadtxt(os.path.join(path,'00%_Athens_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_athens)
plt.show()
train_athens=train_athens[500:6000]

train_kocaeli = np.loadtxt(os.path.join(path,'00%_Kocaeli_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_kocaeli)
plt.show()
train_kocaeli=train_kocaeli[1000:5000]

train_meinung = np.loadtxt(os.path.join(path,'00%_16kaoshiung_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_meinung)
plt.show()
train_meinung=train_meinung[13500:22500]

train_602nantou = np.loadtxt(os.path.join(path,'00%_602nantou_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_602nantou)
plt.show()
train_602nantou=train_602nantou[3500:12500]

train_christchurch11 = np.loadtxt(os.path.join(path,'00%_10newzealand_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_christchurch11)
plt.show()
train_christchurch11=train_christchurch11[500:3000]

train_california19 = np.loadtxt(os.path.join(path,'00%_19california_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_california19)
plt.show()
train_california19=train_california19[2500:10000]
'''
# 10% noises
path='./10% noises'
train_jiji = np.loadtxt(os.path.join(path,'10%noise_8F_dr3%_chichi.dat'))
plt.plot(train_jiji)    
plt.show()
train_jiji=train_jiji[1500:7500,:]

train_norica = np.loadtxt(os.path.join(path,'10%_norica_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_norica)    
plt.show()
train_norica=train_norica[2250:4500,:]

train_mexico = np.loadtxt(os.path.join(path,'10%_mexico_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_mexico)    
plt.show()
train_mexico=train_mexico[6000:13500,:]

train_berkeley = np.loadtxt(os.path.join(path,'10%_berkeley_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_berkeley)
plt.show()
train_berkeley=train_berkeley[2900:3900,:]

train_christchurch = np.loadtxt(os.path.join(path,'10%_11newzealand_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_christchurch)
plt.show()
train_christchurch=train_christchurch[1500:7500,:]

train_jp311 = np.loadtxt(os.path.join(path,'10%_jp311_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_jp311)
plt.show()
train_jp311=train_jp311[5000:22500,:]

train_hualien19 = np.loadtxt(os.path.join(path,'10%_hualien2019_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_hualien19)
plt.show()
train_hualien19=train_hualien19[4500:8000,:]

train_amberley = np.loadtxt(os.path.join(path,'10%_amberley_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_amberley)
plt.show()
train_amberley=train_amberley[7500:40500,:]

train_327nantou = np.loadtxt(os.path.join(path,'10%_nantou0327_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_327nantou)
plt.show()
train_327nantou=train_327nantou[5500:10500,:]

train_1031hualien = np.loadtxt(os.path.join(path,'10%_hualien1031_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_1031hualien)
plt.show()
train_1031hualien=train_1031hualien[3750:12500,:]

train_california20 = np.loadtxt(os.path.join(path,'10%_20california_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_california20)
plt.show()
train_california20=train_california20[2800:5500,:]

train_331hualien = np.loadtxt(os.path.join(path,'10%_hualien331_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_331hualien)
plt.show()
train_331hualien=train_331hualien[3500:9000,:]

train_chile = np.loadtxt(os.path.join(path,'10%_chile_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_chile)
plt.show()
train_chile=train_chile[6250:11000,:]

train_chiayi = np.loadtxt(os.path.join(path,'10%_chiayi_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_chiayi)
plt.show()
train_chiayi=train_chiayi[5500:9000,:]

train_yilan = np.loadtxt(os.path.join(path,'10%_yilan_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_yilan)
plt.show()
train_yilan=train_yilan[3500:6000,:]

train_rome = np.loadtxt(os.path.join(path,'10%_Rome_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_rome)
plt.show()
train_rome=train_rome[1000:4000,:]

train_sakaria = np.loadtxt(os.path.join(path,'10%_Sakaria_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_sakaria)
plt.show()

train_athens = np.loadtxt(os.path.join(path,'10%_Athens_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_athens)
plt.show()
train_athens=train_athens[500:6000,:]

train_kocaeli = np.loadtxt(os.path.join(path,'10%_Kocaeli_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_kocaeli)
plt.show()
train_kocaeli=train_kocaeli[1000:5000,:]

train_meinung = np.loadtxt(os.path.join(path,'10%_meinung_responses_1_8DOF_500Hz_a.dat'))
train_meinung=train_meinung[13500:22500,:]
plt.plot(train_meinung)
plt.show()

train_602nantou = np.loadtxt(os.path.join(path,'10%_nantou0602_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_602nantou)
plt.show()
train_602nantou=train_602nantou[3500:12500,:]

train_christchurch11 = np.loadtxt(os.path.join(path,'10%_10newzealand_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_christchurch11)
plt.show()
train_christchurch11=train_christchurch11[500:3000,:]

train_california19 = np.loadtxt(os.path.join(path,'10%_19california_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_california19)
plt.show()
train_california19=train_california19[2500:10000,:]

# 5% noises
path='./5% noises'
train_jiji = np.loadtxt(os.path.join(path,'./05%_chichi_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_jiji)    
plt.show()
train_jiji=train_jiji[1500:7500,:]

train_norica = np.loadtxt(os.path.join(path,'05%_noricaitaly_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_norica)    
plt.show()
train_norica=train_norica[2250:4500,:]

train_mexico = np.loadtxt(os.path.join(path,'05%_mexico_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_mexico)    
plt.show()
train_mexico=train_mexico[6000:13500,:]

train_berkeley = np.loadtxt(os.path.join(path,'05%_berkeley_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_berkeley)
plt.show()
train_berkeley=train_berkeley[2900:3900,:]

train_christchurch = np.loadtxt(os.path.join(path,'05%_christchurch_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_christchurch)
plt.show()
train_christchurch=train_christchurch[1500:7500,:]

train_jp311 = np.loadtxt(os.path.join(path,'05%_jp311_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_jp311)
plt.show()
train_jp311=train_jp311[5000:22500,:]

train_hualien19 = np.loadtxt(os.path.join(path,'05%_hualien2019_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_hualien19)
plt.show()
train_hualien19=train_hualien19[4500:8000,:]

train_amberley = np.loadtxt(os.path.join(path,'05%_amberley_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_amberley)
plt.show()
train_amberley=train_amberley[7500:40500,:]

train_327nantou = np.loadtxt(os.path.join(path,'05%_nantou327_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_327nantou)
plt.show()
train_327nantou=train_327nantou[5500:10500,:]

train_1031hualien = np.loadtxt(os.path.join(path,'05%_hualien1031_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_1031hualien)
plt.show()
train_1031hualien=train_1031hualien[3750:12500,:]

train_california20 = np.loadtxt(os.path.join(path,'05%_20california_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_california20)
plt.show()
train_california20=train_california20[2800:5500,:]

train_331hualien = np.loadtxt(os.path.join(path,'05%_hualien331_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_331hualien)
plt.show()
train_331hualien=train_331hualien[3500:9000,:]

train_chile = np.loadtxt(os.path.join(path,'05%_chile_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_chile)
plt.show()
train_chile=train_chile[6250:11000,:]

train_chiayi = np.loadtxt(os.path.join(path,'05%_chiayi_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_chiayi)
plt.show()
train_chiayi=train_chiayi[5500:9000,:]

train_yilan = np.loadtxt(os.path.join(path,'05%_yilan_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_yilan)
plt.show()
train_yilan=train_yilan[3500:6000,:]

train_rome = np.loadtxt(os.path.join(path,'05%_Rome_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_rome)
plt.show()
train_rome=train_rome[1000:4000,:]

train_sakaria = np.loadtxt(os.path.join(path,'05%_Sakaria_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_sakaria)
plt.show()

train_athens = np.loadtxt(os.path.join(path,'05%_Athens_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_athens)
plt.show()
train_athens=train_athens[500:6000,:]

train_kocaeli = np.loadtxt(os.path.join(path,'05%_Kocaeli_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_kocaeli)
plt.show()
train_kocaeli=train_kocaeli[1000:5000,:]

train_meinung = np.loadtxt(os.path.join(path,'05%_meinung_responses_1_8DOF_500Hz_a.dat'))
train_meinung=train_meinung[13500:22500,:]
plt.plot(train_meinung)
plt.show()

train_602nantou = np.loadtxt(os.path.join(path,'05%_nantou602_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_602nantou)
plt.show()
train_602nantou=train_602nantou[3500:12500,:]

train_christchurch11 = np.loadtxt(os.path.join(path,'05%_10newzealand_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_christchurch11)
plt.show()
train_christchurch11=train_christchurch11[500:3000,:]

train_california19 = np.loadtxt(os.path.join(path,'05%_19california_responses_1_8DOF_500Hz_a.dat'))
plt.plot(train_california19)
plt.show()
train_california19=train_california19[2500:10000,:]
'''
#%% Relatice Acc (If Absolute Acc, then skip this step)
for i in range(train_jiji.shape[1]-1):
    train_jiji[:,i]=train_jiji[:,i]-train_jiji[:,8]
    train_norica[:,i]=train_norica[:,i]-train_norica[:,8]
    train_mexico[:,i]=train_mexico[:,i]-train_mexico[:,8]
    train_berkeley[:,i]=train_berkeley[:,i]-train_berkeley[:,8]
    train_christchurch[:,i]=train_christchurch[:,i]-train_christchurch[:,8]
    train_jp311[:,i]=train_jp311[:,i]-train_jp311[:,8]
    train_hualien19[:,i]=train_hualien19[:,i]-train_hualien19[:,8]
    train_amberley[:,i]=train_amberley[:,i]-train_amberley[:,8]
    train_327nantou[:,i]=train_327nantou[:,i]-train_327nantou[:,8]
    train_1031hualien[:,i]=train_1031hualien[:,i]-train_1031hualien[:,8]
    train_california20[:,i]=train_california20[:,i]-train_california20[:,8]
    train_331hualien[:,i]=train_331hualien[:,i]-train_331hualien[:,8]
    train_chile[:,i]=train_chile[:,i]-train_chile[:,8]
    train_chiayi[:,i]=train_chiayi[:,i]-train_chiayi[:,8]
    train_yilan[:,i]=train_yilan[:,i]-train_yilan[:,8]
    train_rome[:,i]=train_rome[:,i]-train_rome[:,8]
    train_sakaria[:,i]=train_sakaria[:,i]-train_sakaria[:,8]
    train_athens[:,i]=train_athens[:,i]-train_athens[:,8]
    train_kocaeli[:,i]=train_kocaeli[:,i]-train_kocaeli[:,8]
    train_meinung[:,i]=train_meinung[:,i]-train_meinung[:,8]
    train_602nantou[:,i]=train_602nantou[:,i]-train_602nantou[:,8]
    train_christchurch11[:,i]=train_christchurch11[:,i]-train_christchurch11[:,8]
    train_california19[:,i]=train_california19[:,i]-train_california19[:,8]
#%% Denoise
# chichi
for i in range(train_jiji.shape[0]):
    jiji_temp=decrease_noises(train_jiji[i], time_steps=5)
    train_jiji[i,:len(jiji_temp)]=jiji_temp
# norica
for i in range(train_norica.shape[0]):
    norica_temp=decrease_noises(train_norica[i], time_steps=5)
    train_norica[i,:len(norica_temp)]=norica_temp
# mexico
for i in range(train_mexico.shape[0]):
    mexico_temp=decrease_noises(train_mexico[i], time_steps=5)
    train_mexico[i,:len(mexico_temp)]=mexico_temp
# berkeley
for i in range(train_berkeley.shape[0]):
    berkeley_temp=decrease_noises(train_berkeley[i], time_steps=5)
    train_berkeley[i,:len(berkeley_temp)]=berkeley_temp
# christchurch
for i in range(train_christchurch.shape[0]):
    christchurch_temp=decrease_noises(train_christchurch[i], time_steps=5)
    train_christchurch[i,:len(christchurch_temp)]=christchurch_temp
# 311japan
for i in range(train_jp311.shape[0]):
    jp311_temp=decrease_noises(train_jp311[i], time_steps=5)
    train_jp311[i,:len(jp311_temp)]=jp311_temp
# hualien2019
for i in range(train_hualien19.shape[0]):
    hualien19_temp=decrease_noises(train_hualien19[i], time_steps=5)
    train_hualien19[i,:len(hualien19_temp)]=hualien19_temp
# nantou2013
for i in range(train_327nantou.shape[0]):
    nantou327_temp=decrease_noises(train_327nantou[i], time_steps=5)
    train_327nantou[i,:len(nantou327_temp)]=nantou327_temp
# amberley
for i in range(train_amberley.shape[0]):
    amberley_temp=decrease_noises(train_amberley[i], time_steps=5)
    train_amberley[i,:len(amberley_temp)]=amberley_temp
# hualien2013
for i in range(train_1031hualien.shape[0]):
    hualien1031_temp=decrease_noises(train_1031hualien[i], time_steps=5)
    train_1031hualien[i,:len(hualien1031_temp)]=hualien1031_temp
# california2020
for i in range(train_california20.shape[0]):
    california20_temp=decrease_noises(train_california20[i], time_steps=5)
    train_california20[i,:len(california20_temp)]=california20_temp
# 331 hualien
for i in range(train_331hualien.shape[0]):
    hualien331_temp=decrease_noises(train_331hualien[i], time_steps=5)
    train_331hualien[i,:len(hualien331_temp)]=hualien331_temp
# chile
for i in range(train_chile.shape[0]):
    chile_temp=decrease_noises(train_chile[i], time_steps=5)
    train_chile[i,:len(chile_temp)]=chile_temp
# chiayi
for i in range(train_chiayi.shape[0]):
    chiayi_temp=decrease_noises(train_chiayi[i], time_steps=5)
    train_chiayi[i,:len(chiayi_temp)]=chiayi_temp
# yilan
for i in range(train_yilan.shape[0]):
    yilan_temp=decrease_noises(train_yilan[i], time_steps=5)
    train_yilan[i,:len(yilan_temp)]=yilan_temp
# rome
for i in range(train_rome.shape[0]):
    rome_temp=decrease_noises(train_rome[i], time_steps=5)
    train_rome[i,:len(rome_temp)]=rome_temp
# sakaria
for i in range(train_sakaria.shape[0]):
    sakaria_temp=decrease_noises(train_sakaria[i], time_steps=5)
    train_sakaria[i,:len(sakaria_temp)]=sakaria_temp
# athens
for i in range(train_athens.shape[0]):
    athens_temp=decrease_noises(train_athens[i], time_steps=5)
    train_athens[i,:len(athens_temp)]=athens_temp
# kocaeli
for i in range(train_kocaeli.shape[0]):
    kocaeli_temp=decrease_noises(train_kocaeli[i], time_steps=5)
    train_kocaeli[i,:len(kocaeli_temp)]=kocaeli_temp
# meinung
for i in range(train_meinung.shape[0]):
    meinung_temp=decrease_noises(train_meinung[i], time_steps=5)
    train_meinung[i,:len(meinung_temp)]=meinung_temp
# 602nantou
for i in range(train_602nantou.shape[0]):
    nantou602_temp=decrease_noises(train_602nantou[i], time_steps=5)
    train_602nantou[i,:len(nantou602_temp)]=nantou602_temp
# christchurch 2011
for i in range(train_christchurch11.shape[0]):
    christchurch_temp=decrease_noises(train_christchurch11[i], time_steps=5)
    train_christchurch11[i,:len(christchurch_temp)]=christchurch_temp
# california 2019
for i in range(train_california19.shape[0]):
    california19_temp=decrease_noises(train_california19[i], time_steps=5)
    train_california19[i,:len(california19_temp)]=california19_temp
    
#%% Normalization
jiji=train_jiji/np.max(abs(train_jiji))
norica=train_norica/np.max(abs(train_norica))
mexico=train_mexico/np.max(abs(train_mexico))
berkeley=train_berkeley/np.max(abs(train_berkeley))
christchurch=train_christchurch/np.max(abs(train_christchurch))
jp311=train_jp311/np.max(abs(train_jp311))
hualien19=train_hualien19/np.max(abs(train_hualien19))
amberley=train_amberley/np.max(abs(train_amberley))
nantou=train_327nantou/np.max(abs(train_327nantou))
hualien=train_1031hualien/np.max(abs(train_1031hualien))
california20=train_california20/np.max(abs(train_california20))
hualien331=train_331hualien/np.max(abs(train_331hualien))
chile=train_chile/np.max(abs(train_chile))
chiayi=train_chiayi/np.max(abs(train_chiayi))
yilan=train_yilan/np.max(abs(train_yilan))
rome=train_rome/np.max(abs(train_rome))
sakaria=train_sakaria/np.max(abs(train_sakaria))
athens=train_athens/np.max(abs(train_athens))
kocaeli=train_kocaeli/np.max(abs(train_kocaeli))
meinung=train_meinung/np.max(abs(train_meinung))
nantou602=train_602nantou/np.max(abs(train_602nantou))
christchurch11=train_christchurch11/np.max(abs(train_christchurch11))
california19=train_california19/np.max(abs(train_california19))

#%% Split Data Ver1
# split sequences by time steps
n_steps = 10
jijiX1, jiji_y = split_multisequence(jiji[:,:8], n_steps-1)
noricaX1, norica_y = split_multisequence(norica[:,:8], n_steps-1)
mexicoX1, mexico_y = split_multisequence(mexico[:,:8], n_steps-1)
berkeleyX1, berkeley_y = split_multisequence(berkeley[:,:8], n_steps-1)
christchurchX1, christchurch_y = split_multisequence(christchurch[:,:8], n_steps-1)
jp311X1, jp311_y = split_multisequence(jp311[:,:8], n_steps-1)
hualien19X1, hualien19_y = split_multisequence(hualien19[:,:8], n_steps-1)
amberleyX1, amberley_y = split_multisequence(amberley[:,:8], n_steps-1)
nantouX1, nantou_y = split_multisequence(nantou[:,:8], n_steps-1)
hualienX1, hualien_y = split_multisequence(hualien[:,:8], n_steps-1)
california20X1, california20_y = split_multisequence(california20[:,:8], n_steps-1)
hualien331X1, hualien331_y = split_multisequence(hualien331[:,:8], n_steps-1)
chileX1, chile_y = split_multisequence(chile[:,:8], n_steps-1)
chiayiX1, chiayi_y = split_multisequence(chiayi[:,:8], n_steps-1)
yilanX1, yilan_y = split_multisequence(yilan[:,:8], n_steps-1)
romeX1, rome_y = split_multisequence(rome[:,:8], n_steps-1)
sakariaX1, sakaria_y = split_multisequence(sakaria[:,:8], n_steps-1)
athensX1, athens_y = split_multisequence(athens[:,:8], n_steps-1)
kocaeliX1, kocaeli_y = split_multisequence(kocaeli[:,:8], n_steps-1)
meinungX1, meinung_y = split_multisequence(meinung[:,:8], n_steps-1)
nantou602X1, nantou602_y = split_multisequence(nantou602[:,:8], n_steps-1)
christchurch11X1, christchurch11_y = split_multisequence(christchurch11[:,:8], n_steps-1)
california19X1, california19_y = split_multisequence(california19[:,:8], n_steps-1)

jijiX2 = split_sequence(jiji[:,8], n_steps)
noricaX2 = split_sequence(norica[:,8], n_steps)
mexicoX2 = split_sequence(mexico[:,8], n_steps)
berkeleyX2 = split_sequence(berkeley[:,8], n_steps)
christchurchX2 = split_sequence(christchurch[:,8], n_steps)
jp311X2 = split_sequence(jp311[:,8], n_steps)
hualien19X2 = split_sequence(hualien19[:,8], n_steps)
amberleyX2 = split_sequence(amberley[:,8], n_steps)
nantouX2 = split_sequence(nantou[:,8], n_steps)
hualienX2 = split_sequence(hualien[:,8], n_steps)
california20X2 = split_sequence(california20[:,8], n_steps)
hualien331X2 = split_sequence(hualien331[:,8], n_steps)
chileX2 = split_sequence(chile[:,8], n_steps)
chiayiX2 = split_sequence(chiayi[:,8], n_steps)
yilanX2 = split_sequence(yilan[:,8], n_steps)
romeX2 = split_sequence(rome[:,8], n_steps)
sakariaX2 = split_sequence(sakaria[:,8], n_steps)
athensX2 = split_sequence(athens[:,8], n_steps)
kocaeliX2 = split_sequence(kocaeli[:,8], n_steps)
meinungX2 = split_sequence(meinung[:,8], n_steps)
nantou602X2 = split_sequence(nantou602[:,8], n_steps)
christchurch11X2 = split_sequence(christchurch11[:,8], n_steps)
california19X2 = split_sequence(california19[:,8], n_steps)
### 23 seismic events
trainX1=np.vstack((jijiX1,noricaX1,mexicoX1,berkeleyX1,christchurchX1,christchurch11X1,jp311X1,
                   hualien19X1,amberleyX1,nantouX1,hualienX1,california20X1,
                   hualien331X1,chileX1,chiayiX1,yilanX1,
                   romeX1,sakariaX1,athensX1,kocaeliX1,meinungX1,nantou602X1,california19X1))
trainX2=np.vstack((jijiX2,noricaX2,mexicoX2,berkeleyX2,christchurchX2,christchurch11X2,jp311X2,
                   hualien19X2,amberleyX2,nantouX2,hualienX2,california20X2,
                   hualien331X2,chileX2,chiayiX2,yilanX2,
                  romeX2,sakariaX2,athensX2,kocaeliX2,meinungX2,nantou602X2,california19X2))
# padding the sequences
special_value = 0

'''
###  10 seismic events
trainX1=np.vstack((jijiX1,noricaX1,mexicoX1,berkeleyX1,christchurch11X1,jp311X1,
                   hualien19X1,nantouX1,hualienX1,california20X1))
                   
trainX2=np.vstack((jijiX2,noricaX2,mexicoX2,berkeleyX2,christchurch11X2,jp311X2,
                   hualien19X2,nantouX2,hualienX2,california20X2))
                   
# padding the sequences
special_value = (np.mean(train_jiji)+np.mean(train_mexico)+np.mean(train_norica)
                 +np.mean(train_berkeley)+np.mean(train_1031hualien)+np.mean(train_327nantou)
                 +np.mean(train_california20)+np.mean(train_hualien19)+np.mean(train_jp311)
                 +np.mean(train_christchurch11))/10

### 15 seismic events
trainX1=np.vstack((jijiX1,noricaX1,mexicoX1,berkeleyX1,christchurchX1,jp311X1,
                   hualien19X1,amberleyX1,nantouX1,hualienX1,california20X1,
                   hualien331X1,chileX1,chiayiX1,yilanX1))
trainX2=np.vstack((jijiX2,noricaX2,mexicoX2,berkeleyX2,christchurchX2,jp311X2,
                   hualien19X2,amberleyX2,nantouX2,hualienX2,california20X2,
                   hualien331X2,chileX2,chiayiX2,yilanX2))
# padding the sequences
special_value = (np.mean(train_jiji)+np.mean(train_mexico)+np.mean(train_norica)
                 +np.mean(train_berkeley)+np.mean(train_1031hualien)+np.mean(train_327nantou)
                 +np.mean(train_california20)+np.mean(train_christchurch)
                 +np.mean(train_hualien19)+np.mean(train_jp311)+np.mean(train_amberley)+
                 np.mean(train_331hualien)+np.mean(train_chile)+np.mean(train_chiayi)+
                 np.mean(train_yilan))/15
'''
trainXpad = np.full((trainX1.shape[0], n_steps, trainX1.shape[2]), fill_value=special_value)
for s, x in enumerate(trainX1):
    seq_len = x.shape[0]
    trainXpad[s, 0:seq_len, :] = x

# concatenate along the third axis
train_X = np.dstack((trainX2, trainXpad))

#%% Split Data Ver2
# split sequences by time steps
n_steps = 10
jijiX1, jiji_y = split_multisequence(jiji[:,:8], n_steps)
noricaX1, norica_y = split_multisequence(norica[:,:8], n_steps)
mexicoX1, mexico_y = split_multisequence(mexico[:,:8], n_steps)
berkeleyX1, berkeley_y = split_multisequence(berkeley[:,:8], n_steps)
christchurchX1, christchurch_y = split_multisequence(christchurch[:,:8], n_steps)
jp311X1, jp311_y = split_multisequence(jp311[:,:8], n_steps)
hualien19X1, hualien19_y = split_multisequence(hualien19[:,:8], n_steps)
amberleyX1, amberley_y = split_multisequence(amberley[:,:8], n_steps)
nantouX1, nantou_y = split_multisequence(nantou[:,:8], n_steps)
hualienX1, hualien_y = split_multisequence(hualien[:,:8], n_steps)
california20X1, california20_y = split_multisequence(california20[:,:8], n_steps)
hualien331X1, hualien331_y = split_multisequence(hualien331[:,:8], n_steps)
chileX1, chile_y = split_multisequence(chile[:,:8], n_steps)
chiayiX1, chiayi_y = split_multisequence(chiayi[:,:8], n_steps)
yilanX1, yilan_y = split_multisequence(yilan[:,:8], n_steps)
romeX1, rome_y = split_multisequence(rome[:,:8], n_steps)
sakariaX1, sakaria_y = split_multisequence(sakaria[:,:8], n_steps)
athensX1, athens_y = split_multisequence(athens[:,:8], n_steps)
kocaeliX1, kocaeli_y = split_multisequence(kocaeli[:,:8], n_steps)
meinungX1, meinung_y = split_multisequence(meinung[:,:8], n_steps)
nantou602X1, nantou602_y = split_multisequence(nantou602[:,:8], n_steps)
christchurch11X1, christchurch11_y = split_multisequence(christchurch11[:,:8], n_steps)
california19X1, california19_y = split_multisequence(california19[:,:8], n_steps)

jijiX2 = split_sequence(jiji[:,8], n_steps)
noricaX2 = split_sequence(norica[:,8], n_steps)
mexicoX2 = split_sequence(mexico[:,8], n_steps)
berkeleyX2 = split_sequence(berkeley[:,8], n_steps)
christchurchX2 = split_sequence(christchurch[:,8], n_steps)
jp311X2 = split_sequence(jp311[:,8], n_steps)
hualien19X2 = split_sequence(hualien19[:,8], n_steps)
amberleyX2 = split_sequence(amberley[:,8], n_steps)
nantouX2 = split_sequence(nantou[:,8], n_steps)
hualienX2 = split_sequence(hualien[:,8], n_steps)
california20X2 = split_sequence(california20[:,8], n_steps)
hualien331X2 = split_sequence(hualien331[:,8], n_steps)
chileX2 = split_sequence(chile[:,8], n_steps)
chiayiX2 = split_sequence(chiayi[:,8], n_steps)
yilanX2 = split_sequence(yilan[:,8], n_steps)
romeX2 = split_sequence(rome[:,8], n_steps)
sakariaX2 = split_sequence(sakaria[:,8], n_steps)
athensX2 = split_sequence(athens[:,8], n_steps)
kocaeliX2 = split_sequence(kocaeli[:,8], n_steps)
meinungX2 = split_sequence(meinung[:,8], n_steps)
nantou602X2 = split_sequence(nantou602[:,8], n_steps)
christchurch11X2 = split_sequence(christchurch11[:,8], n_steps)
california19X2 = split_sequence(california19[:,8], n_steps)

trainX1=np.vstack((jijiX1,noricaX1,mexicoX1,berkeleyX1,christchurchX1,christchurch11X1,jp311X1,
                   hualien19X1,amberleyX1,nantouX1,hualienX1,california20X1,
                   hualien331X1,chileX1,chiayiX1,yilanX1,
                   romeX1,sakariaX1,athensX1,kocaeliX1,meinungX1,nantou602X1,california19X1))
trainX2=np.vstack((jijiX2,noricaX2,mexicoX2,berkeleyX2,christchurchX2,christchurch11X2,jp311X2,
                   hualien19X2,amberleyX2,nantouX2,hualienX2,california20X2,
                   hualien331X2,chileX2,chiayiX2,yilanX2,
                  romeX2,sakariaX2,athensX2,kocaeliX2,meinungX2,nantou602X2,california19X2))

# concatenate along the third axis
train_X = np.dstack((trainX2, trainX1))

#%% save train data
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = train_X.shape[2]
trainX = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features)) 

trainy = np.vstack((jiji_y,norica_y,mexico_y,berkeley_y,christchurch_y,christchurch11_y,
                    jp311_y,hualien19_y,amberley_y,nantou_y,hualien_y,california20_y,
                    hualien331_y,chile_y,chiayi_y,yilan_y,rome_y,sakaria_y,
                    athens_y,kocaeli_y,
                    meinung_y,nantou602_y,california19_y))

'''
## 10 seismic events
trainy = np.vstack((jiji_y,norica_y,mexico_y,berkeley_y,christchurch11_y,
                    jp311_y,hualien19_y,nantou_y,hualien_y,california20_y))
## 15 seismic events
trainy = np.vstack((jiji_y,norica_y,mexico_y,berkeley_y,christchurch_y,
                    jp311_y,hualien19_y,amberley_y,nantou_y,hualien_y,california20_y,
                    hualien331_y,chile_y,chiayi_y,yilan_y))
'''
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)

np.save('trainX_lstm',trainX)
np.save('trainy_lstm',trainy)

#%% for CNN_LSTM
'''
# for cnn lstm
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_seq = np.int(np.sqrt(n_steps))
time_steps = np.int(np.sqrt(n_steps))
n_features = trainX.shape[2]
trainX = trainX.reshape((trainX.shape[0], n_seq, time_steps, n_features))
testX = testX.reshape((testX.shape[0], n_seq, time_steps, n_features))
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)
print('testX:',testX.shape)
print('testy:',testy.shape)
np.save('trainX_cnnlstm',trainX)
np.save('trainy_cnnlstm',trainy)
np.save('testX_cnnlstm(kobe health)',testX)
np.save('testy_cnnlstm(kobe health)',testy)
'''
#%% for convlstm
'''
# for convlstm
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = trainX.shape[2]
n_seq = np.int(np.sqrt(n_steps))
n_steps = np.int(np.sqrt(n_steps))
trainX = trainX.reshape((trainX.shape[0], n_seq, 1, n_steps, n_features))
testX = testX.reshape((testX.shape[0], n_seq, 1, n_steps, n_features))
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)
print('testX:',testX.shape)
print('testy:',testy.shape)
np.save('trainX_convlstm',trainX)
np.save('trainy_convlstm',trainy)
np.save('testX_convlstm(kobe health)',testX)
np.save('testy_convlstm(kobe health)',testy)
'''