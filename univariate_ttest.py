#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:02:06 2020

@author: kmsa
"""

import numpy as np
import os.path as op
import dyconnmap
import pickle
from os import listdir
from datetime import datetime
from tqdm import tqdm #count for loops
import matplotlib.pyplot as plt
from os import mkdir
import pandas as pd
import pdb
from dyconnmap.analytic_signal import analytic_signal
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import scipy
from mne import read_labels_from_annot
from mne.datasets import fetch_fsaverage
from collections import OrderedDict
from pprint import pprint

#%%################################
# Plot mean timeseries across subjects
###################################

# Load timeseries (label_ts)
atlas = 'BAita_timeseries_2005/' #'DK_timeseries_0505/'
dir_ts = '/share/FannyMaster/PythonNew/'+atlas+'Subjects_ts/'

mean_ts = []
max_ts = []
min_ts = []

label_ts = []
for file in tqdm(listdir(dir_ts)):
    with open(dir_ts+file, 'rb') as file:
        label_dict = pickle.load(file)
    #print(label_ts)
    label_ts.append(np.array(list(label_dict['timeseries']))[:,0:10000])
   
#mean and standard deviation across subjects
mean_ts = np.mean(label_ts,axis=0)
sd_ts = np.std(label_ts,axis=0)       

# Plot timeseries
plt.figure()
tp = None
n_ch = 5
for c, i in enumerate(range(n_ch)): #range(len(mean_ts)):
    plt.subplot(n_ch,1,c+1)
    if tp == None:    
        plt.plot(mean_ts[i]+sd_ts[i])
        plt.plot(mean_ts[i])
        plt.plot(mean_ts[i]-sd_ts[i])
    else:
        plt.plot(mean_ts[i][0:tp]+sd_ts[i][0:tp])
        plt.plot(mean_ts[i][0:tp])
        plt.plot(mean_ts[i][0:tp]-sd_ts[i][0:tp])

plt.figure()
plt.plot(mean_ts.T+sd_ts.T)
plt.plot(mean_ts.T)
plt.plot(mean_ts.T-sd_ts.T)

tp = 250
plt.figure()
plt.plot(mean_ts.T[0:tp]+sd_ts.T[0:tp])
plt.plot(mean_ts.T[0:tp])
plt.plot(mean_ts.T[0:tp]-sd_ts.T[0:tp])

#%%###########################################################################
# Function to load connectivity matrices
##############################################################################
def getData(dir_features, dir_y_ID, con_type, partialData = False):
    """
    Parameters
    ----------
    dir_features : string
        Directory path to where the features are saved.
    dir_y_ID : string
        Directory path to where the y-vector can be extracted.
    con_type : string
        The desired connectivity measure.

    Returns
    -------
    X : array of arrays
        Matrix containing a vector with all the features for each subject.
        Dimension (number of subjects)x(number of features).
    y : array
        A vector containing the class-information. 
        Remember: 0 = healty controls, 1 = schizophrenic

    """
   
    # Make directory path and get file    
    file_path = dir_features + '/feature_dict_' + con_type + '.pkl'
    with open(file_path, 'rb') as file:
        feature_dict = pickle.load(file)
    
    # Load csv with y classes
    age_gender_dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
    if partialData:
        age_gender_dat = age_gender_dat[~age_gender_dat['Id'].isin(['D950', 'D935', 'D259', 'D255', 'D247', 'D160'])]
    
    
    X = []
    y = []
    for i, row in age_gender_dat.iterrows():
        X.append(feature_dict['features'][row['Id']])
        y.append(row['Group'])
    X = np.array(X)
    y = 1 - pd.Series(y)
    
    return X, y

##############################################################################
def getEgillX(X):
    ## Extract rois (Egill)
    # Old rois with paracentral, used in report
    rois = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    # New rois with parahippocampal
    #rois =  np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    
    rois = np.array([rois,rois]).reshape(-1,order='F')
    roi_mat = np.outer(rois,rois)
    
    # Calculate correlation    
    roi_vec = []
        
    for i in range(6):
        roi_vec.extend(list(roi_mat[np.triu_indices(len(roi_mat), k=1)]))
    
    Xnew = X[:,np.nonzero(roi_vec)]
    Xnew = np.reshape(Xnew,(len(Xnew),sum(roi_vec)))
    return Xnew

def BAitaSig():
    #dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
    dir1 = r'/share/FannyMaster/PythonNew/Lorenzo_SI_table1.xlsx'
    xls = pd.ExcelFile(dir1)
    roi_vec = []
    n_BAitaSig = [] 
    for i in range(6):
        usecol = [j for j in range(1,13)]
        usecol.extend([j for j in range(15,29)])
        if i == 0:
            skiprows = [j for j in range(29,175)]
            skiprows.extend([13,14])
        elif i == 5:
            skiprows = [j for j in range(0,145)]
            skiprows.extend([158, 159, 174])
        else:
            skiprows = [j for j in range(0,29*i)]
            skiprows1 = [j for j in range(29*(i+1),175)]
            skiprows.extend(skiprows1)
            skiprows.extend([13+29*i, 14+29*i])
            
            
        dat = pd.read_excel(xls, '78 HV vs. 25 SDD', na_values=['-'], usecols = usecol, skiprows = skiprows)
        #dat.keys()    
        roi_mat = abs(dat)>3.485124
        roi_flat = list(np.array(roi_mat)[np.triu_indices(len(roi_mat), k=1)])
        roi_vec.extend(roi_flat)

        n_BAitaSig.extend([sum(roi_flat)])
    return roi_vec, n_BAitaSig
    
##############################################################################
def getBAitaSigX(X):
    ## Extract rois (Brodmann Areas collected as in Di Lorenzo et al.)
    
    roi_vec, n_BAitaSig = BAitaSig()
    
    Xnew = X[:,np.nonzero(roi_vec)]
    Xnew = np.reshape(Xnew,(len(Xnew),sum(roi_vec)))
    return Xnew, n_BAitaSig 

##############################################################################
def getLabels(atlas): #getDKLabels
    """
    Returns
    -------
    A list containing Desikan-Killiany brain areas. Hence each element of the 
    list is a string with the name of one area. 
    """
    
    if atlas == 'DKEgill':
        # Desikan-Killiany Atlas 
        fs_dir = fetch_fsaverage(verbose=True)
        subjects_dir = op.dirname(fs_dir) 
        #fs_dir = '/home/kmsa/mne_data/MNE-fsaverage-data/fsaverage' 
        #subjects_dir = '/home/kmsa/mne_data/MNE-fsaverage-data'
        subject = 'fsaverage'
        
        #parc = 'PALS_B12_Brodmann' 
        parc = 'aparc'
        labels = read_labels_from_annot(subject, parc = parc, hemi='both',
                                         surf_name= 'white', annot_fname = None, regexp = None,
                                         subjects_dir = subjects_dir, verbose = None)
        
        
        labels = labels[:-1] #[] 
        labels = [i.name for i in labels]
        
        #rois = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]
        rois =  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        #rois =  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        rois = np.array([rois, rois]).reshape(-1,order='F')
        
        labels = np.array(labels)[np.nonzero(rois)] 
        return labels
        
    elif atlas == 'DK':
        # Desikan-Killiany Atlas 
        fs_dir = fetch_fsaverage(verbose=True)
        subjects_dir = op.dirname(fs_dir) 
        #fs_dir = '/home/kmsa/mne_data/MNE-fsaverage-data/fsaverage' 
        #subjects_dir = '/home/kmsa/mne_data/MNE-fsaverage-data'
        subject = 'fsaverage'
        
        #parc = 'PALS_B12_Brodmann' 
        parc = 'aparc'
        labels = read_labels_from_annot(subject, parc = parc, hemi='both',
                                         surf_name= 'white', annot_fname = None, regexp = None,
                                         subjects_dir = subjects_dir, verbose = None)
        labels = labels[:-1] 
        return labels
    
    elif atlas == 'BAita':
        ita_labels = ['SMA', 'SPL', 'SFC', 'AFC', 'OFC', 'LFC', #'INS',
                     'LTL', 'ACC_new', 'PCC', 'PHG_new', 'IPL', 'FLC', 'PVC']
        labels = np.array([[i+'-lh',i+'-rh'] for i in ita_labels]).reshape(-1)
        return labels
    
    elif atlas == 'BAitaSig':
        ## Extract rois (Brodmann Areas collected as in Di Lorenzo et al.)
        
        roi_vec, n_BAitaSig = BAitaSig() 
        
        ita_labels = ['SMA', 'SPL', 'SFC', 'AFC', 'OFC', 'LFC', #'INS',
                         'LTL', 'ACC_new', 'PCC', 'PHG_new', 'IPL', 'FLC', 'PVC']
        ita_labels = np.array([[i+'-lh',i+'-rh'] for i in ita_labels]).reshape(-1)
        
        # Get indices of triangular matrix
        x, y = np.triu_indices(len(ita_labels),k=1)
        labels = []
        for i in range(6):
            rois = roi_vec[len(roi_vec)//6*i : len(roi_vec)//6*(i+1)]
            sig_idx = np.nonzero(rois)
            
            labels.append([ita_labels[x[j]]+'_' +ita_labels[y[j]] for j in sig_idx[0]])

        return labels, n_BAitaSig
    
##############################################################################
def getConMatrices(dir_nz_coef_idx, dir_nz_coef_val, wanted_info, clf_types, min_fraction, atlas):
    """
    The main function to get the connectivity matrices for plotting. Gives two
    matrices containing different type of weights. One is the mean feature 
    weight, mWu, and the other is the number of occurences weight, Wcount.

    Parameters
    ----------
    dir_nz_coef_idx : string
        Path to the wanted non-zero coefficients indices.
    dir_nz_coef_val : string
        Path to the wanted non-zero coefficients values.
    wanted_info : string
        Can be 'all', 'bands' or 'both'.
    clf_types : list of strings
        Each string is the name of a wanted classifer. E.g. ['lasso', 'svm']
    min_fraction : float
        The minimum fraction of times a feature have been used.

    Returns
    -------
    cons_Wmu_tot : list of lists
        Each inner list contains a connectivity matrix with muW as weights 
        for a given frequency band.
    cons_Wcount_tot : list of lists
        Each inner list contains a connectivity matrix with countW as weights 
        for a given frequency band.

    """
    # Used frequency bands 
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]    
    
    #Number of elements for each frequency band
    if atlas == 'BAitaSig':
        labels, n_BAitaSig = getLabels(atlas)
    else: 
        labels = getLabels(atlas)
        n_BAitaSig = None
    n_feature_bands = int(((len(labels))*(len(labels)-1))/2)
        
    idx2 = np.nonzero(idx)
                                  
    # Get band number and real index
    band_index = [band_idx(idx, n_feature_bands, n_BAitaSig) for idx in idx2[0]]
        
    #Get conmatrix all
    for j in range(len(freq_bands)):
        cons_Wmu, cons_Wcount = makeConMatrix(band_index, mu_w, cv_count, j, atlas, wanted_info)
        cons_Wmu_tot.append(cons_Wmu)
        cons_Wcount_tot.append(cons_Wcount)
    
    return cons_Wmu_tot, cons_Wcount_tot

##############################################################################
def band_idx(idx, n_feature_bands, n_BAitaSig=None):
    """
    Parameters
    ----------
    idx : int
        Index number in vector containing several bands.
    n_feature_bands : int
        Number of feature bands.

    Returns
    -------
    A list with two elements:
        band : int
            Band number.
        idx_real : int
            The real index number.
    """
    if n_BAitaSig == None:
        for band in range(6):
            if (idx in range(band*n_feature_bands, (band+1)*n_feature_bands)):
                idx_real = idx % n_feature_bands
                return [band, idx_real]
    else: 
        n_cumsum = [0]
        n_cumsum.extend(np.cumsum(n_BAitaSig))
        for band, cum_val in enumerate(n_cumsum[1:]):
            if (idx < cum_val):
                idx_real = idx - n_cumsum[band]
                return [band, idx_real]

##############################################################################
def significant_connected_areas(pval_idx, labels, n_BAitaSig=None):
    """
    Parameters
    ----------
    dir_nz_coef_idx : string
        Path to the wanted non-zero coefficients indices. 
    min_fraction : float
        The minimum fraction of times a feature have been used.
    labels : list
        A containig strings with names of brain areas, generated by 
        get*Labels().
    wanted_info : string
        Can be 'all', 'bands' or 'both'. 
    clf_types : list of strings
        Each string corresponds to the name of a classifier.
        
    Returns
    -------
    connected_areas : string
        String containing the most connected areas and what band that is used.

    Notes
    -------
    Prints the areas that are connected more then the minimum fraction of 
    times.

    """
    # Used frequency bands 
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    # Initialize dictionary to return
    connected_areas = OrderedDict()
   
    # Get indices of triangular matrix
    x, y = np.triu_indices(len(labels),k=1)  
    
    # Number of elements for each frequency band 
    n_feature_bands = int(((len(labels))*(len(labels)-1))/2)

    #for j, band in enumerate(list(coef_idx_dict.keys())) :
        # coef_idx = coef_idx_dict[band]
        # # Get top recurrent index values
        # coef_idx_vec = pd.DataFrame(pd.core.common.flatten(coef_idx))
        # min_count = len(coef_idx)*min_fraction # Multiply with fraction to gain percentage
        # coef_idx_count = coef_idx_vec[0].value_counts()
        # coef_idx_top = coef_idx_count[coef_idx_count >= min_count].index.tolist()
        
    # Get band number and real index
    band_index = [band_idx(idx, n_feature_bands, n_BAitaSig) for idx in pval_idx[0]]
        
        
    if n_BAitaSig == None:
        #Print names of most connected brain areas
        con_area = [[freq_bands[i[0]], labels[x[i[1]]], labels[y[i[1]]]] for i in band_index]
        connected_areas['all'] = con_area
    else:
        #Print names of most connected brain areas
        con_area = [[freq_bands[i[0]], labels[i[0]][i[1]]] for i in band_index]
        connected_areas['all'] = con_area
            
    return connected_areas

    
#%%##################################################
# T-test of all roi-roi connections 
#####################################################

test = 'ttest' #'MannWhitneyU'

# atlas = ['DK_timeseries_0505', 'DKEgill_timeseries_2705', 'BAita_timeseries_2005', 'BAitaSig_timeseries_2705']
# atlas2 = ['Desikan-Killiany', 'Desikan-Killiany selected rois (Egill)', 'Brodmann', 'Brodmann selected rois (Lorenzo et al.)']
# atlas3 = ['DK', 'DKEgill', 'BAita', 'BAitaSig']
# con_type = ['plv', 'pli', 'lps']


atlas = ['BAita_timeseries_2005', 'BAitaSig_timeseries_2705', 'DKEgill_timeseries_2705'] # , 'DK_timeseries_0505']
atlas2 = ['Brodmann', 'Brodmann selected rois (Lorenzo et al.)', 'Desikan-Killiany selected rois (Egill)'] #, 'Desikan-Killiany']
atlas3 = ['BAita', 'BAitaSig', 'DKEgill'] #, 'DK']
#atlas3 = ['DK', 'DKEgill', 'BAita', 'BAitaSig']
con_type = ['pli']

dir_y_ID = '/share/FannyMaster/PythonNew/Age_Gender.csv'

partialDat = True

for parc in range(len(atlas)):
    dir_features = '/share/FannyMaster/PythonNew/' + atlas[parc] + '/DiLorenzo/Features/'
    print('------------------------------------')
    print(atlas2[parc])
    print('------------------------------------')
    
    if atlas3[parc] == 'BAitaSig':
        labels, n_BAitaSig = getLabels(atlas3[parc])
    else:
        labels = getLabels(atlas3[parc])
        n_BAitaSig = None
        
    for con in range(len(con_type)):
        
        if atlas[parc]=='DK_timeseries_0505' and con_type[con]=='lps':
            continue
            
        print(con_type[con])
        
        X, y = getData(dir_features, dir_y_ID, con_type[con], partialData = partialDat)

        if atlas[parc] == 'DKEgill_timeseries_2705':
            X = getEgillX(X)        
        elif atlas[parc] == 'BAitaSig_timeseries_2705':
            X, _ = getBAitaSigX(X)

        hc = X[y==1]
        sz = X[y==0]
        
        if test == 'MannWhitneyU':
            pval = []
            for i in range(len(hc[0])):
                stats, pval_tmp = scipy.stats.mannwhitneyu(hc[:,i],sz[:,i])
                pval.append(pval_tmp)
            pval = np.array(pval)
            
        elif test == 'ttest':
            #hc = np.log(hc)
            #sz = np.log(sz)
            stats, pval = ttest_ind(hc,sz,axis=0,equal_var=False)
            
        elif test == 'logF':
            F = np.var(hc[:,i])/np.var(sz[:,i])
            p = scipy.stats.f.sf(stats**2,len(hc[:,i])-1,len(sz[:,i])-1)
        
        print('Number of significant connections:                 ', sum(pval<0.05), ', Minimum p-value:', np.min(pval))
        print('Number of significant connections after Bonferroni:', sum(pval<0.05/len(hc[0])), ', Minimum corrected p-value:', np.min(pval*len(hc[0])))
        
        reject, pval_cor, alphacSidak, alphacBonf = multipletests(pval,alpha=0.05, method='fdr_bh')
        #print('Number of significant connections after FDR (BH):', sum(pval_cor<0.05), ', Minimum corrected p-value:', np.min(pval_cor))
        
        pval_bonf = pval*len(hc[0])
        pval_idx = np.nonzero(pval_bonf<0.05)
        
        #connections = most_connected_areas(pval_idx, min_fraction, labels, wanted_info, clf_types, n_BAitaSig=None)
        connections = significant_connected_areas(pval_idx, labels, n_BAitaSig)
        pprint(connections)
        
        mu_con_sig = {'avg. HC': np.mean(hc[:,pval_idx],axis=0), 'avg. SZ': np.mean(sz[:,pval_idx],axis=0)}
        pprint(mu_con_sig)
        
        # pval_bh = pval_cor
        # pval_idx = np.nonzero(pval_bh<0.05)
        
        # #connections = most_connected_areas(pval_idx, min_fraction, labels, wanted_info, clf_types, n_BAitaSig=None)
        # connections = significant_connected_areas(pval_idx, labels, n_BAitaSig)
        # pprint(connections)
