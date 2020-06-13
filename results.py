#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:12:02 2020

@author: s153968
"""

# For debugging functions import;
import pdb; 
# And then use following for "stopping" the code.
#pdb.set_trace()
#Use "c", "u", "d" etc for navigation

from os import chdir
chdir(r'/share/FannyMaster/PythonNew')

# []
# {}

#%%###########################################################################
## Example connectivity-matrix plot
## Takes about 10 minutes for coherence
##############################################################################

# from testConnectivity import plotConnectivity 
# from utilsResults import getNewestFolderDate

# # YOU MIGHT WANT TO CHANGE THESE:
# freq_band_type = 'DiLorenzo'
# con_type = 'coherence' # 'coherence'
# wanted_info = 'bands' #'all', 'bands' or 'both'
# #################################

# #Static values
# dir_folders = r'/share/FannyMaster/PythonNew/DK_timeseries_'
# newest_date = getNewestFolderDate(dir_folders)
# dir_ts = dir_folders + newest_date + '/Subjects_ts/'

# dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'

# lower = [1.5, 4.,  8., 12., 20., 30.]
# upper = [4. , 8., 12., 20., 30., 80.]

# sub_nb = 8

# plotConnectivity(dir_ts, dir_save, con_type, lower, upper, sub_nb)

#%%###########################################################################
## Average connectivity-matrix plot
##############################################################################

from get_connectivity import plotAvgConnectivity2
from utilsResults import getNewestFolderDate
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
dir_folders = r'/share/FannyMaster/PythonNew/DKEgill_timeseries_2705'
#newest_date = getNewestFolderDate(dir_folders)
#dir_avg_mat_hc =  dir_folders + newest_date + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_hc.pkl'
#dir_avg_mat_scz = dir_folders + newest_date + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_scz.pkl'
#dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'

dir_avg_mat_hc =  dir_folders + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_hc.pkl'
dir_avg_mat_scz = dir_folders + '/' + freq_band_type + '/Features' + '/avg_' + con_type + '_mat_scz.pkl'
dir_save = dir_folders + '/' + freq_band_type +'/Plots'

title = 'DK-expert: Average ' + con_type.upper() + ' for '

plotAvgConnectivity2(dir_avg_mat_hc, dir_avg_mat_scz, dir_save, freq_band_type, title, 'DKEgill')



#%%###########################################################################
## Extract features
##############################################################################

from utilsResults import getLabels, most_connected_areas,getNewestFolderDate
from pprint import pprint


# YOU MIGHT WANT TO CHANGE THESE:
con_type = 'pli' # 'coherence'
wanted_info = 'bands' #'all', 'bands' or 'both'
min_fraction = 0.1 # Fraction of times a feature have been used
atlas = 'DKEgill' # DKEgill, BAita, BAitaSig
partialData = True
#################################

if partialData == True:
    partialDat = 'Partial/'
else: 
    partialDat = '/'

freq_band_type = 'DiLorenzo'
clf_types = ['lasso']

#Static values
# dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
# newest_date = getNewestFolderDate(dir_folders)
# dir_nz_coef_idx = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/nz_coef_idx_'

dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_2705'
dir_nz_coef_idx = dir_folders + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/nz_coef_idx_'


# Extract atlas labels
if atlas == 'BAitaSig':
    labels, n_BAitaSig = getLabels(atlas)
else:
    labels = getLabels(atlas)
    n_BAitaSig = None
connected_areas = most_connected_areas(dir_nz_coef_idx, min_fraction, labels, wanted_info, clf_types, n_BAitaSig)

pprint(connected_areas['alpha'])


# from utilsResults import getDKLabels, most_connected_areas, getNewestFolderDate
# from pprint import pprint

# # YOU MIGHT WANT TO CHANGE THESE:
# freq_band_type = 'DiLorenzo'
# con_type = 'pli' # 'coherence'
# wanted_info = 'bands' #'all', 'bands' or 'both'
# min_fraction = 0.1 # Fraction of times a feature have been used
# #################################
# clf_types = ['lasso']

# atlas = 'DKEgill' # DKEgill, BAita, BAitaSig

# #Static values
# dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
# newest_date = getNewestFolderDate(dir_folders)
# dir_nz_coef_idx = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() + '/nz_coef_idx_'

# # Extract atlas labels
# labels = getDKLabels()

# connected_areas = most_connected_areas(dir_nz_coef_idx, min_fraction, labels, wanted_info, clf_types)

# pprint(connected_areas)


#%%###########################################################################
## Box plots of auc results
##############################################################################

from utilsResults import boxplots_auc
from utilsResults import getNewestFolderDate

# YOU MIGHT WANT TO CHANGE THESE:
con_types = ['plv', 'pli', 'lps']
wanted_info = 'bands' #'all', 'bands' or 'both'
atlas = 'DKEgill' # DKEgill, BAita, BAitaSig
partialData = True
#################################

if partialData == True:
    partialDat = 'Partial'
else: 
    partialDat = ''

freq_band_type = 'DiLorenzo'
auc_types = ['lasso']
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
newest_date = getNewestFolderDate(dir_folders)

#Static values
for con_type in con_types:
    dir_auc = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + '/' + con_type.capitalize() + '/auc_'
    
    dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots' + partialDat
    
    figsize=(10,6)
    
    fig = boxplots_auc(dir_auc, dir_save, auc_types, wanted_info, figsize, con_type, atlas)


#%%###########################################################################
## Connections on the brain plots
##############################################################################
from utilsResults import getConMatrices, plotConnections
from utilsResults import getNewestFolderDate

# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'pli' # 'coherence'
wanted_info = 'bands' #'all', 'bands' or 'both'
# Fraction of times a feature have been used
min_fraction = 0.1
atlas = 'DKEgill' # DKEgill, BAita, BAitaSig
#################################

#Static values
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_nz_coef_idx = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() + '/nz_coef_idx_'
dir_nz_coef_val = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() + '/nz_coef_val_'


clf_types = ['lasso']
cons_Wmu_tot, cons_Wcount_tot = getConMatrices(dir_nz_coef_idx, dir_nz_coef_val, wanted_info, clf_types, min_fraction, atlas)

plotConnections(cons_Wmu_tot, atlas + ' - Weights by mean features - ' + con_type, atlas)
plotConnections(cons_Wcount_tot, atlas + ' - Weights by occurences - ' + con_type, atlas)


#%%###########################################################################
## Brain plots with labels
##############################################################################
from matplotlib import cm
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from utilsResults import getCoordinates, getLabels
atlas = 'BAita'
coordinates1 = getCoordinates(atlas)
cons_W_zero = np.zeros((len(coordinates1), len(coordinates1)))

if atlas == 'BAita':
    suptitle = 'DL: Di Lorenzo et al. inspired atlas'
elif atlas == 'DKEgill':
    suptitle = 'DK-expert: Expert inspired atlas'

fig, ax = plt.subplots(1,2, figsize= (12, 7.5))
for i in range(2):
    if i == 0:
        disp_mode = 'z'
        title = 'Top View'
    else:
        disp_mode = 'y'
        title = 'Back View'
    fig1 = plotting.plot_connectome(cons_W_zero, coordinates1, display_mode=disp_mode,
                                     node_color='k', node_size=4, annotate = True, 
                                     axes = ax[i]) # ,title= 'ita')
    fig1.annotate(size = 23)
    marker_color = list(cm.Paired(np.linspace(0,1, (len(coordinates1)//2-1))))
    marker_color.extend([np.array([0,0,0,1])])
    labels = getLabels(atlas) #['SMA', 'SPL','SFC', 'AFC', 'OFC', 'LFC', #'INS', 'LTL','ACC', 'PCC', 'PHG', 'IPL','FLC','PVC']
    plt.title(title, fontsize = 20)
    for i,j in enumerate(range(0, len(coordinates1), 2)):
        fig1.add_markers(marker_coords = [coordinates1[j], coordinates1[j+1]], 
                         marker_size = 200, marker_color = marker_color[i], 
                         label= labels[j].split('-')[0])
        # fig1.add_markers(marker_coords = [coordinates1[j+1]], marker_size = 3, 
        #              marker_color = marker_color[i], 
        #              label= marker_color[i])
plt.legend(bbox_to_anchor = (1.05,1), borderaxespad = 0., loc = 'upper left',
           prop= {'size': 18})

plt.suptitle(suptitle, fontsize = 25)

plt.savefig('/share/FannyMaster/PythonNew/Figures/' + atlas + '_brainplot.jpg', bbox_inches = 'tight')



#%%###########################################################################
## Permutation test histograms
##############################################################################
from utilsResults import getPermResults
import pickle
import matplotlib.pyplot as plt
from utilsResults import getNewestFolderDate

# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'pli' # 'coherence'
wanted_info = 'bands' #'all', 'bands' or 'both'
nb_perms = 100
atlas = 'BAita' # DKEgill, BAita, BAitaSig
partialData = False
#################################

if partialData == True:
    partialDat = 'Partial/'
else: 
    partialDat = '/'

#Statis values
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'

newest_date = getNewestFolderDate(dir_folders)
dir_auc =      dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/auc_'
dir_auc_perm = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/perm_auc_'

dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'+partialDat


clf_types = ['lasso']

if atlas == 'BAitaSig':
    atlas = 'DL-sig'
elif atlas == 'BAita':
    atlas = 'DL'
else: 
    atlas = 'DK-expert'
    
title = atlas +": Permutation of 100 mean AUC's - " + con_type.upper()

getPermResults(dir_auc, dir_auc_perm, dir_save, wanted_info, nb_perms, title, con_type)

#%%###########################################################################
## Plot significant brain connections
##############################################################################
from utilsResults import plotSigBrainConnections, getConMatrices, getNewestFolderDate, getPermResults
import numpy as np
# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'pli' # 'coherence'
wanted_info = 'bands' #'all', 'bands' or 'both'
nb_perms = 100
# Fraction of times a feature have been used
min_fraction = 0.1
atlas = 'BAita' # DKEgill, BAita, BAitaSig
partialData = True
#################################

clf_types = ['lasso']

if partialData == True:
    partialDat = 'Partial/'
    denom = 260
else: 
    partialDat = '/'
    denom = 280
    
title = atlas +": Permutation of 100 mean AUC's - " + con_type.upper()

dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'

newest_date = getNewestFolderDate(dir_folders)
dir_auc =      dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/auc_'
dir_auc_perm = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/perm_auc_'
dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'+partialDat


title = atlas +": Permutation of 100 mean AUC's - " + con_type.upper()
pval_list = getPermResults(dir_auc, dir_auc_perm, dir_save, wanted_info, nb_perms, title, con_type)

if np.min(pval_list) < 0.05:
    #Static values
    dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
    newest_date = getNewestFolderDate(dir_folders)
    dir_nz_coef_idx = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/nz_coef_idx_'
    dir_nz_coef_val = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/nz_coef_val_'
    
    cons_Wmu_tot, cons_Wcount_tot = getConMatrices(dir_nz_coef_idx, dir_nz_coef_val, wanted_info, clf_types, min_fraction, atlas)
    
    for i, pval in enumerate(pval_list):
        if pval < 0.05:
            cons_Wmu = cons_Wmu_tot[i]
            cons_Wcount = cons_Wcount_tot[i]
            plotSigBrainConnections(cons_Wmu, cons_Wcount, atlas, i, dir_save, con_type, denom)
else: 
    print('---------------------------')
    print('No p-value lower then 0.05')



#%%###########################################################################
## Plot Di Lorenzo et al.'s significant brain connections 
##############################################################################
from utilsResults import plotItalianBrainConnections2, plotItalianBrainConnections, getItalianSigMatrices, getNewestFolderDate
import numpy as np
# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'pli' # 'coherence'
wanted_info = 'bands' #'all', 'bands' or 'both'
nb_perms = 100
# Fraction of times a feature have been used
min_fraction = 0.1
atlas = 'BAita' # DKEgill, BAita, BAitaSig
partialData = True
#################################

clf_types = ['lasso']

if partialData == True:
    partialDat = 'Partial/'
    denom = 260
else: 
    partialDat = '/'
    denom = 280
    
title = atlas +": Permutation of 100 mean AUC's - " + con_type.upper()

dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'

newest_date = getNewestFolderDate(dir_folders)
dir_auc =      dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/auc_'
dir_auc_perm = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/perm_auc_'
dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'+partialDat


# title = atlas +": Permutation of 100 mean AUC's - " + con_type.upper()
# pval_list = getPermResults(dir_auc, dir_auc_perm, dir_save, wanted_info, nb_perms, title, con_type)


itaW = getItalianSigMatrices()
band_idx = [1,2]
#plotItalianBrainConnections2(itaW[band_idx[0]].to_numpy(), itaW[band_idx[1]].to_numpy(), band_idx, dir_save)

plotItalianBrainConnections(itaW[3].to_numpy(), 3, dir_save)

itaW2 = abs(itaW[3])>5
itaW2 = itaW[3][itaW2.fillna(0)]
plotItalianBrainConnections2(itaW[3].to_numpy(), itaW2, 3, dir_save)

# if np.min(pval_list) < 0.05:
#     #Static values
#     dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
#     newest_date = getNewestFolderDate(dir_folders)
#     dir_nz_coef_idx = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/nz_coef_idx_'
#     dir_nz_coef_val = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/nz_coef_val_'
    
#     cons_Wmu_tot, cons_Wcount_tot = getConMatrices(dir_nz_coef_idx, dir_nz_coef_val, wanted_info, clf_types, min_fraction, atlas)
    
#     for i, pval in enumerate(pval_list):
#         if pval < 0.05:
#             cons_Wmu = cons_Wmu_tot[i]
#             cons_Wcount = cons_Wcount_tot[i]
#             plotSigBrainConnections(cons_Wmu, cons_Wcount, atlas, i, dir_save, con_type, denom)
# else: 
#     print('---------------------------')
#     print('No p-value lower then 0.05')

#%%###########################################################################
## Compare connectivity methods
##############################################################################
from makeClassificationTest2 import getData
from utilsResults import getNewestFolderDate
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

X1, _ = getData(dir_features, dir_y_ID, con_type1)
X2, _ = getData(dir_features, dir_y_ID, con_type2)


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
from makeClassificationTest2 import getData
from utilsResults import getNewestFolderDate
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

X1, _ = getData(dir_features, dir_y_ID, con_type1)
X2, _ = getData(dir_features, dir_y_ID, con_type2)


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
from makeClassificationTest2 import getData
from utilsResults import getNewestFolderDate
import matplotlib.pyplot as plt
from sklearn import preprocessing

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
    
    X1, _ = getData(dir_features, dir_y_ID, con1)
    X2, _ = getData(dir_features, dir_y_ID, con2)
    
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

#%%###########################################################################
## Plot AUC for all bands
##############################################################################

from glob import glob
import os.path as op
from utilsResults import getNewestFolderDate
import pickle
import matplotlib.pyplot as plt
# Parameters to change:
freq_band_type = 'DiLorenzo'
con_types = ['plv', 'pli', 'lps']
#####################################

atlas = 'DKEgill' # DKEgill, BAita, BAitaSig

# Directories
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
newest_date = getNewestFolderDate(dir_folders)


freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
#freq_bands = ["all"]

for band in freq_bands:
    figsize=(10,6)
    auc_plot = []
    x_label = []
    for con_type in con_types:
        dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResultsPartial/' + con_type.capitalize()
        file_paths = glob(dir_save + '/auc_lasso' + '_bands*.pkl')
        newest_file = max(file_paths, key=op.getctime)
        with open(newest_file, 'rb') as file:
            auc = pickle.load(file)
        auc_plot.append(auc[band])
        x_label.append(con_type.capitalize())
        
    fig = plt.figure(figsize=figsize)
    plt.boxplot(auc_plot)
    locs, junk = plt.xticks()
    plt.xticks(locs, x_label)
    plt.ylabel('AUC')
    plt.title('Boxplots over the ' + band + ' band for ' + str(len(auc[band])) + ' loops')
    # plt.axhline(y=0.5) #, linestyle = 'dashed', lw=1, color='white')
    plt.show()
    
    #dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots'
    #fig.savefig(dir_save + '/Boxplots_connectivity_' + band + '_auc.jpg', bbox_inches = 'tight')




