# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 10:22:30 2020

@author: CSHuangLab
"""

# %% import modules
import numpy as np
from matplotlib import pyplot as plt
import random
import os
SEED=1996
random.seed(SEED)
np.random.seed(SEED)
# %% define function
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

# %% data preprocessing
# load data
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
'''
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
train_jiji_scaled=train_jiji/np.max(abs(train_jiji))
train_norica_scaled=train_norica/np.max(abs(train_norica))
train_mexico_scaled=train_mexico/np.max(abs(train_mexico))
train_berkeley_scaled=train_berkeley/np.max(abs(train_berkeley))
train_christchurch_scaled=train_christchurch/np.max(abs(train_christchurch))
train_jp311_scaled=train_jp311/np.max(abs(train_jp311))
train_hualien19_scaled=train_hualien19/np.max(abs(train_hualien19))
train_amberley_scaled=train_amberley/np.max(abs(train_amberley))
train_nantou_scaled=train_327nantou/np.max(abs(train_327nantou))
train_hualien_scaled=train_1031hualien/np.max(abs(train_1031hualien))
train_california20_scaled=train_california20/np.max(abs(train_california20))
train_hualien331_scaled=train_331hualien/np.max(abs(train_331hualien))
train_chile_scaled=train_chile/np.max(abs(train_chile))
train_chiayi_scaled=train_chiayi/np.max(abs(train_chiayi))
train_yilan_scaled=train_yilan/np.max(abs(train_yilan))
train_rome_scaled=train_rome/np.max(abs(train_rome))
train_sakaria_scaled=train_sakaria/np.max(abs(train_sakaria))
train_athens_scaled=train_athens/np.max(abs(train_athens))
train_kocaeli_scaled=train_kocaeli/np.max(abs(train_kocaeli))
train_meinung_scaled=train_meinung/np.max(abs(train_meinung))
train_nantou602_scaled=train_602nantou/np.max(abs(train_602nantou))
train_christchurch11_scaled=train_christchurch11/np.max(abs(train_christchurch11))
train_california19_scaled=train_california19/np.max(abs(train_california19))

# %% split the data (time steps)
# specify degree of freeedom
while True:
    try:
        n=int(input('please entry the number of d.o.f:'))# number of elements
        a = list(map(int,input("\nEnter the numbers : ").strip().split()))[:n]# read inputs from user using map() function 
        dof_arr=np.array(a)
        print("the measured floors are-", a)
        break
    except:
        print("please check the errors...there must be a space between numbers you entry")
# horizontally stack columns
jiji=[]
norica=[]
mexico=[]
berkeley=[]
christchurch=[]
jp311=[]
hualien19=[]
amberley=[]
nantou=[]
hualien=[]
california20=[]
hualien331=[]
chile=[]
chiayi=[]
yilan=[]
rome=[]
sakaria=[]
athens=[]
kocaeli=[]
meinung=[]
nantou602=[]
christchurch11=[]
california19=[]
for i in dof_arr:
    print(i)
    jiji.append(train_jiji_scaled[:,7-i+1])
    norica.append(train_norica_scaled[:,7-i+1])
    mexico.append(train_mexico_scaled[:,7-i+1])
    berkeley.append(train_berkeley_scaled[:,7-i+1])
    christchurch.append(train_christchurch_scaled[:,7-i+1])
    jp311.append(train_jp311_scaled[:,7-i+1])
    hualien19.append(train_hualien19_scaled[:,7-i+1])
    amberley.append(train_amberley_scaled[:,7-i+1])
    nantou.append(train_nantou_scaled[:,7-i+1])
    hualien.append(train_hualien_scaled[:,7-i+1])
    california20.append(train_california20_scaled[:,7-i+1])
    hualien331.append(train_hualien331_scaled[:,7-i+1])
    chile.append(train_chile_scaled[:,7-i+1])
    chiayi.append(train_chiayi_scaled[:,7-i+1])
    yilan.append(train_yilan_scaled[:,7-i+1])
    rome.append(train_rome_scaled[:,7-i+1])
    sakaria.append(train_sakaria_scaled[:,7-i+1])
    athens.append(train_athens_scaled[:,7-i+1])
    kocaeli.append(train_kocaeli_scaled[:,7-i+1])
    meinung.append(train_meinung_scaled[:,7-i+1])
    nantou602.append(train_nantou602_scaled[:,7-i+1])
    christchurch11.append(train_christchurch11_scaled[:,7-i+1])
    california19.append(train_california19_scaled[:,7-i+1])
jiji=np.array(jiji).T
norica=np.array(norica).T
mexico=np.array(mexico).T
berkeley=np.array(berkeley).T
christchurch=np.array(christchurch).T
jp311=np.array(jp311).T
hualien19=np.array(hualien19).T
amberley=np.array(amberley).T
nantou=np.array(nantou).T
hualien=np.array(hualien).T
california20=np.array(california20).T
hualien331=np.array(hualien331).T
chile=np.array(chile).T
chiayi=np.array(chiayi).T
yilan=np.array(yilan).T
rome=np.array(rome).T
sakaria=np.array(sakaria).T
athens=np.array(athens).T
kocaeli=np.array(kocaeli).T
meinung=np.array(meinung).T
nantou602=np.array(nantou602).T
christchurch11=np.array(christchurch11).T
california19=np.array(california19).T

# split sequences by time steps
n_steps = 60
jijiX1, jiji_y = split_multisequence(jiji, n_steps-1)
noricaX1, norica_y = split_multisequence(norica, n_steps-1)
mexicoX1, mexico_y = split_multisequence(mexico, n_steps-1)
berkeleyX1, berkeley_y = split_multisequence(berkeley, n_steps-1)
christchurchX1, christchurch_y = split_multisequence(christchurch, n_steps-1)
jp311X1, jp311_y = split_multisequence(jp311, n_steps-1)
hualien19X1, hualien19_y = split_multisequence(hualien19, n_steps-1)
amberleyX1, amberley_y = split_multisequence(amberley, n_steps-1)
nantouX1, nantou_y = split_multisequence(nantou, n_steps-1)
hualienX1, hualien_y = split_multisequence(hualien, n_steps-1)
california20X1, california20_y = split_multisequence(california20, n_steps-1)
hualien331X1, hualien331_y = split_multisequence(hualien331, n_steps-1)
chileX1, chile_y = split_multisequence(chile, n_steps-1)
chiayiX1, chiayi_y = split_multisequence(chiayi, n_steps-1)
yilanX1, yilan_y = split_multisequence(yilan, n_steps-1)
romeX1, rome_y = split_multisequence(rome, n_steps-1)
sakariaX1, sakaria_y = split_multisequence(sakaria, n_steps-1)
athensX1, athens_y = split_multisequence(athens, n_steps-1)
kocaeliX1, kocaeli_y = split_multisequence(kocaeli, n_steps-1)
meinungX1, meinung_y = split_multisequence(meinung, n_steps-1)
nantou602X1, nantou602_y = split_multisequence(nantou602, n_steps-1)
christchurch11X1, christchurch11_y = split_multisequence(christchurch11, n_steps-1)
california19X1, california19_y = split_multisequence(california19, n_steps-1)

jijiX2 = split_sequence(train_jiji_scaled[:,8], n_steps)
noricaX2 = split_sequence(train_norica_scaled[:,8], n_steps)
mexicoX2 = split_sequence(train_mexico_scaled[:,8], n_steps)
berkeleyX2 = split_sequence(train_berkeley_scaled[:,8], n_steps)
christchurchX2 = split_sequence(train_christchurch_scaled[:,8], n_steps)
jp311X2 = split_sequence(train_jp311_scaled[:,8], n_steps)
hualien19X2 = split_sequence(train_hualien19_scaled[:,8], n_steps)
amberleyX2 = split_sequence(train_amberley_scaled[:,8], n_steps)
nantouX2 = split_sequence(train_nantou_scaled[:,8], n_steps)
hualienX2 = split_sequence(train_hualien_scaled[:,8], n_steps)
california20X2 = split_sequence(train_california20_scaled[:,8], n_steps)
hualien331X2 = split_sequence(train_hualien331_scaled[:,8], n_steps)
chileX2 = split_sequence(train_chile_scaled[:,8], n_steps)
chiayiX2 = split_sequence(train_chiayi_scaled[:,8], n_steps)
yilanX2 = split_sequence(train_yilan_scaled[:,8], n_steps)
romeX2 = split_sequence(train_rome_scaled[:,8], n_steps)
sakariaX2 = split_sequence(train_sakaria_scaled[:,8], n_steps)
athensX2 = split_sequence(train_athens_scaled[:,8], n_steps)
kocaeliX2 = split_sequence(train_kocaeli_scaled[:,8], n_steps)
meinungX2 = split_sequence(train_meinung_scaled[:,8], n_steps)
nantou602X2 = split_sequence(train_nantou602_scaled[:,8], n_steps)
christchurch11X2 = split_sequence(train_christchurch11_scaled[:,8], n_steps)
california19X2 = split_sequence(train_california19_scaled[:,8], n_steps)

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
trainXpad = np.full((trainX1.shape[0], n_steps, trainX1.shape[2]), fill_value=special_value)
for s, x in enumerate(trainX1):
    seq_len = x.shape[0]
    trainXpad[s, 0:seq_len, :] = x
# concatenate along the third axis
train_X = np.dstack((trainX2, trainXpad))

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = train_X.shape[2]
trainX = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features)) 

trainy = np.vstack((jiji_y,norica_y,mexico_y,berkeley_y,christchurch_y,christchurch11_y,
                    jp311_y,hualien19_y,amberley_y,nantou_y,hualien_y,california20_y,
                    hualien331_y,chile_y,chiayi_y,yilan_y,rome_y,sakaria_y,
                    athens_y,kocaeli_y,
                    meinung_y,nantou602_y,california19_y))

print('trainX:',trainX.shape)
print('trainy:',trainy.shape)


np.save('trainX_lstm_incomplete',trainX)
np.save('trainy_lstm_incomplete',trainy)


#%%
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
#%%
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