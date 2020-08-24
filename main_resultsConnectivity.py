#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:12:02 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

import pdb; #For debugging add pdb.set_trace() in function use c for continue, u for up, exit for exiting debug mode etc.

from os import chdir
chdir(r'/share/rsEEG/Scripts_Fanny/')

# []
# {}


#%%###########################################################################
## Average connectivity-matrix plot
##############################################################################

from utils_resultsConnectivity import plotAvgConnectivity2
from utils_joint import getNewestFolderDate
# sns.reset_orig()
# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'lps' #['plv', 'pli', 'lps']
####################################


###################################################
# Brodmann Italians significant/all
###################################################
dir_folders = r'/share/FannyMaster/PythonNew/BAitaSig_timeseries_' # Same as BAita
newest_date = getNewestFolderDate(dir_folders)
dir_avg_mat_hc =  dir_folders + newest_date + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_hc.pkl'
dir_avg_mat_scz = dir_folders + newest_date + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_scz.pkl'
dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'

title = 'DL: Average ' + con_type.upper() + ' for '

plotAvgConnectivity2(dir_avg_mat_hc, dir_avg_mat_scz, dir_save, freq_band_type, title, 'BAita')
#plotAvgConnectivity(dir_avg_mat_scz, dir_save, freq_band_type, title_scz)    

###################################################
# Desikan-Killiany, Egill areas
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_avg_mat_hc =  dir_folders + newest_date + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_hc.pkl'
dir_avg_mat_scz = dir_folders + newest_date + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_scz.pkl'
dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'

title = 'DK-expert: Average ' + con_type.upper() + ' for '

plotAvgConnectivity2(dir_avg_mat_hc, dir_avg_mat_scz, dir_save, freq_band_type, title, 'DKEgill')

#%%###########################################################################
## t-statistics matrix plot
##############################################################################

from utils_resultsConnectivity import plotTstatConnectivity
from utils_joint import getNewestFolderDate
import pickle
import numpy as np
# sns.reset_orig()
# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'lps' #['plv', 'pli', 'lps']
atlas = 'DKEgill'
####################################

if atlas == 'BAita':
    title = 'DL: Comparison of SCZ vs. HC - ' + con_type
elif atlas == 'DKEgill':
    title = 'DK-expert: Comparison of SCZ vs. HC - ' + con_type
    
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_Tstat = dir_folders + newest_date + '/' + freq_band_type + '/FeaturesNew/t_stat_' + con_type + '.pkl'
dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'
plotTstatConnectivity(dir_Tstat, dir_save, freq_band_type, title)  



#%%###########################################################################
## Compare connectivity methods
##############################################################################
from utils_joint import getNewestFolderDate, get_Xy
import matplotlib.pyplot as plt

# Parameters to change:
freq_band_type = 'DiLorenzo'
con_type1 = 'plv' # 'corr' or 'coherence'
con_type2 = 'lps' # 'corr' or 'coherence'
subject_nb =  [1, 2, 3, 40, 60, 80]
fig, ax = plt.subplots(2,3)
#####################################

# Directories
dir_folders = r'/share/FannyMaster/PythonNew/BAita_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'

X1, _ = get_Xy(dir_features, dir_y_ID, con_type1)
X2, _ = get_Xy(dir_features, dir_y_ID, con_type2)


for i in range(len(subject_nb)): 
    subject = subject_nb[i]
    ax[i//3][i%3].scatter(X1[subject_nb], X2[subject_nb])
    ax[i//3][i%3].set_xlabel(con_type1.capitalize())
    ax[i//3][i%3].set_ylabel(con_type2.capitalize())
    ax[i//3][i%3].set_title('Subject ' + str(subject))
plt.tight_layout()

#fig.suptitle('Scatter plot for subject ' + str(subject_nb))
plt.show()


#%%###########################################################################
## Compare two connectivity methods for several subjects
##############################################################################
from utils_joint import getNewestFolderDate, get_Xy
import matplotlib.pyplot as plt

# Parameters to change:
freq_band_type = 'DiLorenzo'
con_type1 = 'corr' # 'corr' or 'coherence'
con_type2 = 'pli' # 'corr' or 'coherence'
roi_nb =  [1, 100, 500, 1000, 1500, 2000]
fig, ax = plt.subplots(2,3)
#####################################

# Directories
dir_folders = r'/share/FannyMaster/PythonNew/DK_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'

X1, _ = get_Xy(dir_features, dir_y_ID, con_type1)
X2, _ = get_Xy(dir_features, dir_y_ID, con_type2)


for i in range(len(roi_nb)): 
    roi= roi_nb[i]
    ax[i//3][i%3].scatter(X1[:, roi], X2[:,roi])
    ax[i//3][i%3].set_xlabel(con_type1.capitalize())
    ax[i//3][i%3].set_ylabel(con_type2.capitalize())
    ax[i//3][i%3].set_title('ROI ' + str(roi))
plt.tight_layout()

#fig.suptitle('Scatter plot for subject ' + str(subject_nb))
plt.show()
#fig = plt.figure(figsize=figsize)


#%%###########################################################################
## Compare all connectivity methods for a given subject
##############################################################################
from utils_joint import getNewestFolderDate, get_Xy
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

# Parameters to change:
freq_band_type = 'DiLorenzo'
con_types = ['plv', 'pli', 'lps']
subject_nb =  75
fig, ax = plt.subplots(1,3, figsize = (10,3.3)) # See len(pairs) 
atlas = 'BAitaSig' # DKEgill, BAita, BAitaSig
#####################################
pairs = [(r2, r1) for r1 in range(len(con_types)) for r2 in range(r1)]



# Directories
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas +'_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 
#dir_features = dir_folders + '/' + freq_band_type + '/Features' 
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'

# scaler_out = preprocessing.StandardScaler().fit(X_tr)
# X_tr =  scaler_out.transform(X_tr)
sns.set()
count = 0
for pair in pairs: 
    con1 = con_types[pair[0]]
    con2 = con_types[pair[1]]
    
    X1, _ = get_Xy(dir_features, dir_y_ID, con1)
    X2, _ = get_Xy(dir_features, dir_y_ID, con2)
    
    scaler_out = preprocessing.StandardScaler().fit(X1)
    X1 = scaler_out.transform(X1)
    scaler_out = preprocessing.StandardScaler().fit(X2)
    X2 = scaler_out.transform(X2)
    
    # ax[count//3][count%3].scatter(X1[subject_nb], X2[subject_nb])
    # ax[count//3][count%3].set_xlabel(con1.capitalize())
    # ax[count//3][count%3].set_ylabel(con2.capitalize())
    # ax[count//3][count%3].set_title('')
    sns.scatterplot(x = X1[subject_nb], y = X2[subject_nb], ax = ax[count])
    # ax[count].scatter(X1[subject_nb], X2[subject_nb])
    ax[count].set_xlabel(con1.capitalize())
    ax[count].set_ylabel(con2.capitalize())
    ax[count].set_title('')
    count += 1
plt.tight_layout(pad = 1)
fig.subplots_adjust(top=0.9)

fig.suptitle('Shows the relation of two different connectivity measures for one subject ' , fontsize=16)
plt.show()

dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'
fig.savefig(dir_save + '/Connectivity_relations.jpg', bbox_inches = 'tight')
sns.reset_orig()




