# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:29:55 2020

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
#%% relative acc
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

# %% split the data (time steps)
# split sequences by time steps
n_steps = 12
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

# reshape from [samples, timesteps] into [samples, timesteps, features]
jijiX2 = np.squeeze(jijiX2)
noricaX2 = np.squeeze(noricaX2)
mexicoX2 = np.squeeze(mexicoX2)
berkeleyX2 = np.squeeze(berkeleyX2)
christchurchX2 = np.squeeze(christchurchX2)
jp311X2 = np.squeeze(jp311X2)
hualien19X2 = np.squeeze(hualien19X2)
amberleyX2 = np.squeeze(amberleyX2)
nantouX2 = np.squeeze(nantouX2)
hualienX2 = np.squeeze(hualienX2)
california20X2 = np.squeeze(california20X2)
hualien331X2 = np.squeeze(hualien331X2)
chileX2=np.squeeze(chileX2)
chiayiX2=np.squeeze(chiayiX2)
yilanX2=np.squeeze(yilanX2)

romeX2=np.squeeze(romeX2)
sakariaX2=np.squeeze(sakariaX2)
athensX2=np.squeeze(athensX2)
kocaeliX2=np.squeeze(kocaeliX2)
meinungX2=np.squeeze(meinungX2)
nantou602X2=np.squeeze(nantou602X2)
christchurch11X2=np.squeeze(christchurch11X2)
california19X2=np.squeeze(california19X2)

# flatten input
n_input_temp = jijiX1.shape[1] * jijiX1.shape[2]
jijiX1 = jijiX1.reshape((jijiX1.shape[0], n_input_temp))
noricaX1 = noricaX1.reshape((noricaX1.shape[0], n_input_temp))
mexicoX1 = mexicoX1.reshape((mexicoX1.shape[0], n_input_temp))
berkeleyX1 = berkeleyX1.reshape((berkeleyX1.shape[0], n_input_temp))
christchurchX1 = christchurchX1.reshape((christchurchX1.shape[0], n_input_temp))
jp311X1 = jp311X1.reshape((jp311X1.shape[0], n_input_temp))
hualien19X1 = hualien19X1.reshape((hualien19X1.shape[0], n_input_temp))
amberleyX1 = amberleyX1.reshape((amberleyX1.shape[0], n_input_temp))
nantouX1 = nantouX1.reshape((nantouX1.shape[0], n_input_temp))
hualienX1 = hualienX1.reshape((hualienX1.shape[0], n_input_temp))
california20X1 = california20X1.reshape((california20X1.shape[0], n_input_temp))
hualien331X1 = hualien331X1.reshape((hualien331X1.shape[0], n_input_temp))
chileX1 = chileX1.reshape((chileX1.shape[0], n_input_temp))
chiayiX1 = chiayiX1.reshape((chiayiX1.shape[0], n_input_temp))
yilanX1 = yilanX1.reshape((yilanX1.shape[0], n_input_temp))
romeX1 = romeX1.reshape((romeX1.shape[0], n_input_temp))
sakariaX1 = sakariaX1.reshape((sakariaX1.shape[0], n_input_temp))
athensX1 = athensX1.reshape((athensX1.shape[0], n_input_temp))
kocaeliX1 = kocaeliX1.reshape((kocaeliX1.shape[0], n_input_temp))
meinungX1 = meinungX1.reshape((meinungX1.shape[0], n_input_temp))
nantou602X1 = nantou602X1.reshape((nantou602X1.shape[0], n_input_temp))
christchurch11X1 = christchurch11X1.reshape((christchurch11X1.shape[0], n_input_temp))
california19X1 = california19X1.reshape((california19X1.shape[0], n_input_temp))

# concatenate along columns to assemble trainX, testX
jiji_X = np.concatenate((jijiX1, jijiX2), axis=1)
norica_X = np.concatenate((noricaX1, noricaX2), axis=1)
mexico_X = np.concatenate((mexicoX1, mexicoX2), axis=1)
berkeley_X = np.concatenate((berkeleyX1, berkeleyX2), axis=1)
christchurch_X = np.concatenate((christchurchX1, christchurchX2), axis=1)
jp311_X = np.concatenate((jp311X1, jp311X2), axis=1)
hualien19_X = np.concatenate((hualien19X1, hualien19X2), axis=1)
amberley_X = np.concatenate((amberleyX1, amberleyX2), axis=1)
nantou_X = np.concatenate((nantouX1, nantouX2), axis=1)
hualien_X = np.concatenate((hualienX1, hualienX2), axis=1)
california20_X = np.concatenate((california20X1, california20X2), axis=1)
hualien331_X = np.concatenate((hualien331X1, hualien331X2), axis=1)
chile_X = np.concatenate((chileX1, chileX2), axis=1)
chiayi_X = np.concatenate((chiayiX1, chiayiX2), axis=1)
yilan_X = np.concatenate((yilanX1, yilanX2), axis=1)
rome_X = np.concatenate((romeX1, romeX2), axis=1)
sakaria_X = np.concatenate((sakariaX1, sakariaX2), axis=1)
athens_X = np.concatenate((athensX1, athensX2), axis=1)
kocaeli_X = np.concatenate((kocaeliX1, kocaeliX2), axis=1)
meinung_X = np.concatenate((meinungX1, meinungX2), axis=1)
nantou602_X = np.concatenate((nantou602X1, nantou602X2), axis=1)
christchurch11_X = np.concatenate((christchurch11X1, christchurch11X2), axis=1)
california19_X = np.concatenate((california19X1, california19X2), axis=1)

# 23 seismic events
trainX = np.vstack((jiji_X,norica_X,mexico_X,berkeley_X,christchurch_X,christchurch11_X,jp311_X,
                    hualien19_X,amberley_X,nantou_X,hualien_X,california20_X,hualien331_X,
                    chile_X,chiayi_X,yilan_X,rome_X,sakaria_X,athens_X,kocaeli_X,meinung_X,
                    nantou602_X,california19_X))
trainy = np.vstack((jiji_y,norica_y,mexico_y,berkeley_y,christchurch_y,christchurch11_y,jp311_y,
                    hualien19_y,amberley_y,nantou_y,hualien_y,california20_y,hualien331_y,
                    chile_y,chiayi_y,yilan_y,rome_y,sakaria_y,athens_y,kocaeli_y,meinung_y,
                    nantou602_y,california19_y))

'''
# 10 seismic events
trainX = np.vstack((jiji_X,norica_X,mexico_X,berkeley_X,christchurch11_X,jp311_X,hualien19_X,nantou_X,hualien_X,california20_X))
trainy = np.vstack((jiji_y,norica_y,mexico_y,berkeley_y,christchurch11_y,jp311_y,hualien19_y,nantou_y,hualien_y,california20_y))
# 15 seismic evetns
trainX = np.vstack((jiji_X,norica_X,mexico_X,berkeley_X,christchurch_X,jp311_X,
                    hualien19_X,amberley_X,nantou_X,hualien_X,california20_X,hualien331_X,
                    chile_X,chiayi_X,yilan_X))
trainy = np.vstack((jiji_y,norica_y,mexico_y,berkeley_y,christchurch_y,jp311_y,
                    hualien19_y,amberley_y,nantou_y,hualien_y,california20_y,hualien331_y,
                    chile_y,chiayi_y,yilan_y))
'''
n_input = trainX.shape[1]
print('trainX:',trainX.shape)
print('trainy:',trainy.shape)
np.save('trainX_mlp',trainX)
np.save('trainy_mlp',trainy)
