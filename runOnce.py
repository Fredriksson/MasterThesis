#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:01:27 2020

@author: s153968
"""
from os import chdir
chdir(r'/share/FannyMaster/PythonNew')

# []
# {}

#%%###########################################################################
## Extract Desikan-Killiany timeseries from preprocessed data
##############################################################################

from extract_timeseries import extract_ts
from datetime import datetime
#import mkdir

atlas = 'DKEgill'

#dir_prepro_dat = r'P:\PC Glostrup\Lukkede Mapper\Forskningsenhed\CNSR\Fanny\PreprocessedData290420\PreprocessedData'
dir_prepro_dat = r'/share/FannyMaster/PythonNew/Preprocessed'

# Frequency bands
lower = [1.5, 4.,  8., 12., 20., 30.]
upper = [4. , 8., 12., 20., 30., 80.]

#dir_save = r'P:\PC Glostrup\Lukkede Mapper\Forskningsenhed\CNSR\Fanny\Desikan-Killiany_Timeseries\Subjects'+'\\'

date = datetime.now().strftime("%d%m")
dir_save = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_' + date
if not op.exists(dir_save):
    mkdir(dir_save)
    print('\nCreated new path : ' + dir_save)
dir_save += '/Subjects_ts'


# Make the extraction and save the time series
extract_ts(dir_prepro_dat, dir_save, lower, upper, atlas)


#%%###########################################################################
## Make connectivity vectors from time series 
##############################################################################

from get_connectivity import getConnectivity  
from utilsResults import getNewestFolderDate
# Parameters to change:
freq_band_type = 'DiLorenzo'
con_types = ['pli', 'plv', 'lps']
###################################

# Directories
dir_folders = r'/share/FannyMaster/PythonNew/BA_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_ts = dir_folders + newest_date + '/Subjects_ts/'
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
dir_save = dir_folders + newest_date + '/' + freq_band_type + '/Features'

# Di Lorenzo frequency bands
lower = [1.5, 4.,  8., 12., 20., 30.] # If you change this, change freq_band_type
upper = [4. , 8., 12., 20., 30., 80.] # If you change this, change freq_band_type


for con_type in con_types:
    getConnectivity(dir_ts, dir_y_ID, dir_save, con_type, lower, upper)

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
## NEW - Make connectivity vectors from time series  
##############################################################################

from get_connectivity import getConnectivity2 
from utilsResults import getNewestFolderDate
# Parameters to change:
freq_band_type = 'DiLorenzo'
con_types = ['pli', 'plv', 'lps']
atlas = ['DKEgill', 'BAita']

###################################

for atl in atlas:
    # Directories
    dir_folders = r'/share/FannyMaster/PythonNew/' + atl + '_timeseries_2705'
    dir_ts = dir_folders + '/Subjects_ts/'
    dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
    dir_save = dir_folders + '/' + freq_band_type + '/Features2'
    
    # Di Lorenzo frequency bands
    lower = [1.5, 4.,  8., 12., 20., 30.] # If you change this, change freq_band_type
    upper = [4. , 8., 12., 20., 30., 80.] # If you change this, change freq_band_type
    
    
    for con_type in con_types:
        getConnectivity2(dir_ts, dir_y_ID, dir_save, con_type, lower, upper)

#%%###########################################################################
## Make classification, and permutation
##############################################################################

# from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
# from sklearn.linear_model import Lasso
# from utilsResults import getNewestFolderDate

# # Parameters to change:
# freq_band_type = 'DiLorenzo'
# con_type = 'corr' #'lps_plv' # 'corr' or 'coherence'
# #####################################


# # Directories
# dir_folders = r'/share/FannyMaster/PythonNew/DK_timeseries_'
# newest_date = getNewestFolderDate(dir_folders)
# dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 
# dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
# dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 

# X,y = getData(dir_features, dir_y_ID, con_type)

# n_scz_te = 2
# reps = range(20)
# parameters = {'lasso' : {'alpha': [i*0.01 for i in range(8,14,1)]}}
# classifiers = {'lasso' : Lasso(max_iter = 10000)}
    
# #%% Facts
# nb_loops = len(reps)*(sum(y==0)//n_scz_te)
# test_size = len(y)//(sum(y==0)//n_scz_te)
# print('\nNumber of loops = ' + str(nb_loops) + 
#       '\nTest size = ' + str(test_size) + 
#       '\nNumber of scz in test set = ' + str(n_scz_te))


#%%###########################################################################
#%%
# con_types = ['pli', 'plv', 'coherence', 'corr', 'lps_plv']

# # Directories
# dir_folders = r'/share/FannyMaster/PythonNew/DK_timeseries_'
# newest_date = getNewestFolderDate(dir_folders)
# dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 
# dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'

# for con_type in con_types:    
#     dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/Egill_Areas/' + con_type.capitalize() 
#     X,y = getData(dir_features, dir_y_ID, con_type)
#     X = getEgillX(X)
#     separate_bands = False # False = All bands together
#     parameters = getEgillParameters(con_type, separate_bands)
#     perms = range(1) # 1 = No permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters)
    
#     separate_bands = True # True = The bands are seperated
#     parameters = getEgillParameters(con_type, separate_bands)
#     perms = range(1) # 1 = no permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters)

# print('\nStarted with permutations')

# for con_type in con_types:    
#     dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
#     X,y = getData(dir_features, dir_y_ID, con_type)
#     X = getEgillX(X)
#     separate_bands = False # False = All bands together
#     parameters = getEgillParameters(con_type, separate_bands)
#     perms = range(100) # 1 = No permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters)
    
#     separate_bands = True # True = The bands are seperated
#     parameters = getEgillParameters(con_type, separate_bands)
#     perms = range(100) # 1 = no permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters)
    
#%%###########################################################################
 


#%%###########################################################################
#%%
# from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters
# con_types = ['pli', 'plv', 'lps_plv']


# # Directories
# dir_folders = r'/share/FannyMaster/PythonNew/BAitaSig_timeseries_'
# newest_date = getNewestFolderDate(dir_folders)
# dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 
# dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'


# n_scz_te = 2
# reps = range(20)
# classifiers = {'lasso' : Lasso(max_iter = 10000)}


# for con_type in con_types:    
#     dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
#     X,y = getData(dir_features, dir_y_ID, con_type)
#     X, n_BAitaSig = getBAitaSigX(X)
#     separate_bands = False # False = All bands together
    
#     parameters = getBAitaSigParameters(con_type, separate_bands)
    
#     perms = range(1) # 1 = No permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters, n_BAitaSig)
    
#     separate_bands = True # True = The bands are seperated
    
#     parameters = getBAitaSigParameters(con_type, separate_bands)
    
#     perms = range(1) # 1 = no permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters, n_BAitaSig)

# print('\nStarted with permutations')

# for con_type in con_types:    
#     dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
#     X,y = getData(dir_features, dir_y_ID, con_type)
#     X, n_BAitaSig = getBAitaSigX(X)
#     separate_bands = False # False = All bands together
#     parameters = getBAitaSigParameters(con_type, separate_bands)
#     perms = range(100) # 1 = No permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters, n_BAitaSig)
    
#     separate_bands = True # True = The bands are seperated
#     parameters = getBAitaSigParameters(con_type, separate_bands)
#     perms = range(100) # 1 = no permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters, n_BAitaSig)
#%%###########################################################################
# No permutations 
from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

partialData = False

con_types = ['pli', 'lps']
freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}

###################################################
# Brodmann Italians significant
###################################################
dir_folders = r'/share/FannyMaster/PythonNew/BAitaSig_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type)
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
    perms = range(1) # 1 = no permutations
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
    X,y = getData(dir_features, dir_y_ID, con_type)
    
    # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getBAitaParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getBAitaParameters(con_type, separate_bands)
    perms = range(1) # 1 = no permutations
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
    X,y = getData(dir_features, dir_y_ID, con_type)
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
    perms = range(1) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)


#%%###########################################################################
# Permutations
##############################################################################



###########################################################################
###########################################################################
###########################################################################
###########################################################################
## RUN THIS!!!!!
###########################################################################
###########################################################################
###########################################################################
###########################################################################

from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters


freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}


# ###################################################
# # Brodmann Italians significant
# ###################################################
# dir_folders = r'/share/FannyMaster/PythonNew/BAitaSig_timeseries_'
# newest_date = getNewestFolderDate(dir_folders)
# dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 



# for con_type in con_types:    
#     dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
#     X,y = getData(dir_features, dir_y_ID, con_type)
#     X, n_BAitaSig = getBAitaSigX(X)
    
#     # # All bands together
#     # separate_bands = False # False = All bands together
#     # parameters = getBAitaSigParameters(con_type, separate_bands)
#     # perms = range(100) # 1 = No permutations
#     # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#     #               classifiers, parameters, n_BAitaSig)
#     # Separate bands
#     separate_bands = True # True = The bands are seperated
#     parameters = getBAitaSigParameters(con_type, separate_bands)
#     perms = range(100) # 1 = no permutations
#     CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#                   classifiers, parameters, n_BAitaSig)
   
        
###################################################
# Brodmann Italians all
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/BAita_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['plv']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getBAitaParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getBAitaParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)

###################################################
# Desikan-Killiany, Egill areas
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['lps']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData=True)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)


dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['plv']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)



# Times ten

#%% Separate bands, no permutation
# # Takes about 9 minutes for corr
# # Takes about 6 minutes for coherence
# separate_bands = True # True = The bands are seperated
# perms = range(1) # 1 = no permutations
# CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#               classifiers, parameters)

# #%% All bands together, permutation
# # Takes about 10 hours for coherence
# separate_bands = False # False = All bands together
# num_perm = 100
# perms = range(num_perm) # Makes num_perm permutations
# CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
#               classifiers, parameters)

# #%% Separate bands, permutation
# separate_bands = True # True = The bands are seperated
# num_perm = 100 
# perms = range(num_perm) # Makes num_perm permutations
# CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save,
#               classifiers, parameters)

##############################################################################
##############################################################################


#%%
#%%###########################################################################
# No permutations 
from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

partialData = True

con_types = ['plv']
freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}

###################################################
# Brodmann Italians significant
###################################################
dir_folders = r'/share/FannyMaster/PythonNew/BAitaSig_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
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
    perms = range(1) # 1 = no permutations
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
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getBAitaParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getBAitaParameters(con_type, separate_bands)
    perms = range(1) # 1 = no permutations
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
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(1) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)


  
    
#%%
#%%###########################################################################
# Permutations


from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

freq_band_type = 'DiLorenzo'
partialData = True
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}

# To run: BAitaSig: PLI-theta, 
#         BAita: PLI-theta, PLI-beta1, LPS-theta
#         DKEgill: PLI-theta, PLI-alpha, PLI-gamma, LPS-theta, LPS-alpha

###################################################
# Brodmann Italians significant
###################################################
dir_folders = r'/share/FannyMaster/PythonNew/BAitaSig_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['plv']
#specificBands = ['theta']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    X, n_BAitaSig = getBAitaSigX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getBAitaSigParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters, n_BAitaSig)
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getBAitaSigParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters, n_BAitaSig) #, specificBands=specificBands)
   
        
###################################################
# Brodmann Italians all
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/BAita_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['plv']
#specificBands = ['theta']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getBAitaParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getBAitaParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters) #, specificBands=specificBands)
    

###################################################
# Desikan-Killiany, Egill areas
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['plv']
# specificBands = ['theta', 'alpha']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters) #, specificBands=specificBands)



#%%###########################################################################
#%%###########################################################################
#%%######## NEW DK-EGILL
#%%###########################################################################
#%%###########################################################################



###################################################
# DK-Egill - Permutations - Partial Data
################################################### 

from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

freq_band_type = 'DiLorenzo'
partialData = True
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}

dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['pli', 'lps']
# specificBands = ['theta', 'alpha']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters) #, specificBands=specificBands)   



print('###################################################')
print('DONE WITH # DK-Egill - Permutations - Partial Data - LPS and PLI')
print('################################################### ')

###################################################
# DK-Egill - No permutation - Partial Data
###################################################  

from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

partialData = True

con_types = ['pli', 'lps']
freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}


 
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(1) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)

print('###################################################')
print('DONE WITH # DK-Egill - No Permutations - Partial Data - LPS and PLI')
print('################################################### ')






###################################################
# DK-Egill - No Permutations - Full Data
###################################################  

# No permutations 
from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

partialData = False

con_types = ['pli', 'lps']
freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}


 
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type)
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
    perms = range(1) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)
    
print('###################################################')
print('DONE WITH # DK-Egill - No Permutations - Full Data - LPS and PLI')
print('################################################### ')

###################################################
# DK-Egill - Permutations - Full Data
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['pli', 'lps']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type)
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
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)
    

print('###################################################')
print('DONE WITH # DK-Egill - Permutations - Full Data - LPS and PLI')
print('################################################### ')

  ##
##
##
##
##
###################################################
# DK-Egill - Permutations - Partial Data
################################################### 

from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

freq_band_type = 'DiLorenzo'
partialData = True
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}



con_types = ['plv']
# specificBands = ['theta', 'alpha']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(100) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters) #, specificBands=specificBands)   


print('###################################################')
print('DONE WITH # DK-Egill - Permutations - Partial Data - PLV')
print('################################################### ')

###################################################
# DK-Egill - No permutation - Partial Data
###################################################  

from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

partialData = True

con_types = ['plv']
freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}


 
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type, partialData)
    X = getEgillX(X)
    
    # # All bands together
    # separate_bands = False # False = All bands together
    # parameters = getEgillParameters(con_type, separate_bands)
    # perms = range(1) # 1 = No permutations
    # CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
    #               classifiers, parameters)
    
    # Separate bands
    separate_bands = True # True = The bands are seperated
    parameters = getEgillParameters(con_type, separate_bands)
    perms = range(1) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)


print('###################################################')
print('DONE WITH # DK-Egill - No Permutations - Partial Data - PLV')
print('################################################### ')





###################################################
# DK-Egill - No Permutations - Full Data
###################################################  

# No permutations 
from makeClassificationTest2 import CV_classifier, getData, getEgillParameters, getEgillX
from sklearn.linear_model import Lasso
from utilsResults import getNewestFolderDate
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters

partialData = False

con_types = ['plv']
freq_band_type = 'DiLorenzo'

dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(20)
classifiers = {'lasso' : Lasso(max_iter = 10000)}


 
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type)
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
    perms = range(1) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)
    
print('###################################################')
print('DONE WITH # DK-Egill - No Permutations - Full Data - PLV')
print('################################################### ')

###################################################
# DK-Egill - Permutations - Full Data
###################################################   
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 

con_types = ['plv']
for con_type in con_types:    
    dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
    X,y = getData(dir_features, dir_y_ID, con_type)
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
    perms = range(100) # 1 = no permutations
    CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters)  
    
print('###################################################')
print('DONE WITH # DK-Egill - Permutations - Full Data - PLV')
print('################################################### ')   
    
    