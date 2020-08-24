#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:01:27 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

from os import chdir
chdir(r'/share/rsEEG/Scripts_Fanny/')

# []
# {}

#%%###########################################################################
## Extract Desikan-Killiany timeseries from preprocessed data
##############################################################################

from utils_runOnce_sourceLocalization import extract_ts
from datetime import datetime
import os.path as op
from os import mkdir

atlas = 'DKLobes'
pecans = ['1', '2']

for p in pecans:
    dir_prepro_dat = r'/share/rsEEG/Preprocessed/PECANS' + p + '/'
    
    # Frequency bands
    lower = [1.5, 4.,  8., 12., 20., 30.]
    upper = [4. , 8., 12., 20., 30., 80.]
    
    date = datetime.now().strftime("%d%m")
    dir_save = r'/share/rsEEG/Timeseries/PECANS' + p + '/' + atlas + '_timeseries_' + date
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    dir_save += '/Subjects_ts'
    
    
    # Make the extraction and save the time series
    extract_ts(dir_prepro_dat, dir_save, lower, upper, atlas)

# Takes about 3.5 hours

#%%###########################################################################
## Make connectivity vectors from time series 
##############################################################################

from utils_runOnce_connectivity import getConnectivity2 
from utils_joint import getNewestFolderDate
# Parameters to change:
freq_band_type = 'DiLorenzo'
con_types = ['lps']
# con_types = ['pli']
atlas = 'DK'
###################################

pecans = ['1', '2']

for p in pecans:
# Directories
    dir_folders = r'/share/rsEEG/Timeseries/PECANS' + p + '/' + atlas + '_timeseries_'
    newest_date = getNewestFolderDate(dir_folders)
    dir_ts = dir_folders + newest_date + '/Subjects_ts/'
    if p == 1:
        dir_y_ID = r'/share/rsEEG/Scripts_Fanny/Data/Age_Gender.csv'
    else:
        dir_y_ID = r'/share/rsEEG/Scripts_Fanny/Data/Id_Group_red.csv'
        
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/FeaturesNew'
    
    # Di Lorenzo frequency bands
    lower = [1.5, 4.,  8., 12., 20., 30.] # If you change this, change freq_band_type
    upper = [4. , 8., 12., 20., 30., 80.] # If you change this, change freq_band_type
    
    # lower = [1.5]
    # upper = [4.]
    
    for con_type in con_types:
        getConnectivity2(dir_ts, dir_y_ID, dir_save, con_type, lower, upper)

# Brodmann:
# PLV: Takes around 50 minutes
# PLI: Takes around 25 minutes
# LPS: Takes around 42 minutes
# Coherence: Takes around 2.25 hours
# LPS_csd: Takes around 6 minutes

# Desikan-Killiany: (These are old times)
# PLI: Takes about 2.5 hours in total.
# LPS: Takes about 45 minutes.
# Lps_csd: Takes about 4.5 hours. 
# Corr: Takes about 25 minutes in total
# Coherence: Takes about 9-10 minutes per subject -> about 14 hours in total


#%%###########################################################################
# Make Classifications
##############################################################################
from utils_runOnce_classification import CV_classifier, getEgillParameters, getEgillX
from utils_runOnce_classification import getBAitaSigX, getBAitaSigParameters, getBAitaParameters
from utils_joint import get_Xy, getNewestFolderDate
from sklearn.linear_model import Lasso

partialData = False # True or False

con_types = ['pli', 'lps', 'plv']
freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/rsEEG/Scripts_Fanny/Data/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}
perms = range(1) # 1 = no permutations
###################################################
# Brodmann Italians significant
###################################################
dir_folders = r'/share/FannyMaster/PythonNew/BAitaSig_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = get_Xy(dir_features, dir_y_ID, con_type, partialData)
    X, n_BAitaSig = getBAitaSigX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getBAitaSigParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters, n_BAitaSig)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getBAitaSigParameters(con_type, separate_bands)
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters, n_BAitaSig)
    
###################################################
# Brodmann Italians all
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/BAita_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = get_Xy(dir_features, dir_y_ID, con_type, partialData)
    
    # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getBAitaParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getBAitaParameters(con_type, separate_bands)
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)
    

###################################################
# Desikan-Killiany, Egill areas
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = get_Xy(dir_features, dir_y_ID, con_type, partialData)
    X = getEgillX(X)
    
    # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)



    
    