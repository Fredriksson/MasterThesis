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
## Extract features
##############################################################################

from utils_resultsClassification import getLabels, most_connected_areas
from utils_joint import getNewestFolderDate
from pprint import pprint


# YOU MIGHT WANT TO CHANGE THESE:
con_type = 'lps' # 'coherence'
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
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_nz_coef_idx = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + con_type.capitalize() + '/nz_coef_idx_'

# Extract atlas labels
if atlas == 'BAitaSig':
    labels, n_BAitaSig = getLabels(atlas)
else:
    labels = getLabels(atlas)
    n_BAitaSig = None
connected_areas = most_connected_areas(dir_nz_coef_idx, min_fraction, labels, wanted_info, clf_types, n_BAitaSig)

pprint(connected_areas['theta'])


#%%###########################################################################
## Box plots of auc results
##############################################################################

from utils_resultsClassification import boxplots_auc
from utils_joint import getNewestFolderDate

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
# newest_date = '2705'

#Static values
for con_type in con_types:
    dir_auc = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults' + partialDat + '/' + con_type.capitalize() + '/auc_'
    
    dir_save = dir_folders + newest_date + '/' + freq_band_type +'/Plots' + partialDat
    
    figsize=(10,6)
    
    fig = boxplots_auc(dir_auc, dir_save, auc_types, wanted_info, figsize, con_type, atlas)
    
#%%###########################################################################
## Permutation test histograms
##############################################################################
from utils_resultsClassification import getPermResults
import matplotlib.pyplot as plt
from utils_joint import getNewestFolderDate

# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'lps' # 'coherence'
wanted_info = 'bands' #'all', 'bands' or 'both'
nb_perms = 100
atlas = 'DKEgill' # DKEgill, BAita, BAitaSig
partialData = True
#################################

if partialData == True:
    partialDat = 'Partial/'
else: 
    partialDat = '/'

#Statis values
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'

newest_date = getNewestFolderDate(dir_folders)
# newest_date = '2705'
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
## Connections on the brain plots
##############################################################################
from utils_resultsClassification import getConMatrices, plotConnections
from utils_joint import getNewestFolderDate

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
from utils_resultsClassification import getCoordinates, getLabels
atlas = 'DKEgill'
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
## Plot significant brain connections
##############################################################################
from utils_resultsClassification import plotSigBrainConnections, getConMatrices, getPermResults
from utils_joint import getNewestFolderDate
import numpy as np
# YOU MIGHT WANT TO CHANGE THESE:
freq_band_type = 'DiLorenzo'
con_type = 'lps' # 'coherence'
wanted_info = 'bands' #'all', 'bands' or 'both'
nb_perms = 100
# Fraction of times a feature have been used
min_fraction = 0.1
atlas = 'DKEgill' # DKEgill, BAita, BAitaSig
partialData = False
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
from utils_resultsClassification import plotItalianBrainConnections2, plotItalianBrainConnections, getItalianSigMatrices
from utils_joint import getNewestFolderDate
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