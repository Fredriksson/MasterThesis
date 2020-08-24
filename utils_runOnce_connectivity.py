#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:56:16 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

import numpy as np
import os.path as op
import dyconnmap
import pickle
from os import listdir
from tqdm import tqdm #count for loops
from os import mkdir
import pandas as pd
from dyconnmap.analytic_signal import analytic_signal

import pdb #For debugging add pdb.set_trace() in function use c for continue, u for up, exit for exiting debug mode etc.

#{}
#[]

#%%###########################################################################
def getSubjectClass(dir_y_ID):
    '''
    Function that returns a dictionary providing each subject number and the 
    corresonding class. 

    Parameters
    ----------
    dir_y_ID : string
        Path to were the excel file with the subject classes can be found.

    Returns
    -------
    sub_classes : dictionary
        Contains each subject number as keys and the corresponding class.

    '''
    
    sub_classes = {}
    # Load csv with y classes
    age_gender_dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
    # Generate dictionary with subject number and class
    for i in range(len(age_gender_dat)):
        sub_classes[age_gender_dat['Id'][i]] = age_gender_dat['Group'][i]
    
    return sub_classes

##############################################################################
def lps(data, fb, fs):
    '''
    Function that calculates lagged phase synchronisation. It only calcultes 
    the values for the upper triangle. If the denomenator is zero it sets the 
    lps value tozero. Note that this might be a problem for other programs!
    

    Parameters
    ----------
    data : array of arrays
        Data matrix that you would like to calculate the pairwisae lps for.
    fb : 
        DESCRIPTION.
    fs : list
        List with the desired lower and upper limit of the frequency range.

    Returns
    -------
    lps : list of lists
        Matrix with the LPS values of the upper triangle.

    '''    
    # Define the list of pairwise comparisons & allocate space for the matrix
    n_channels, _ = np.shape(data)
    pairs = [(r2, r1) for r1 in range(n_channels) for r2 in range(r1)]
    lps = np.zeros((n_channels, n_channels)) 
    
    # Get the Fourier transformed and normalized data
    _, u_phase, _ = analytic_signal(data, fb, fs) 
    
    # Loop over the different pair combinations
    for pair in pairs:
        # Get the pairs correspnding normalized Fourier transformation
        u1, u2 = u_phase[pair,]
        # Make it independent of phase
        ts_plv = np.exp(1j * (u1-u2))
        
        # Calculate the complex values phase locking value
        r = np.sum(ts_plv) / float(data.shape[1])
        
        # Get numerator and denomenator for LPS
        num = np.power(np.imag(r), 2)
        denom = 1-np.power(np.real(r), 2)
        
        # Make sure to not divide by zero.
        if denom == 0:
            lps[pair] = 0
        else:
            lps[pair] = num/denom

    return lps


##############################################################################
def connectivity(con_type, label_ts, fb, fs):
    """
    Function that calculates the desired functional connectivity. Only specific
    chosen connectivities can be calculated and more could be added.
    
    Parameters
    ----------
    con_type : string
        What connectivity to use.
    label_ts : list
        Time serie for one subject.
    fb : list of lists
        The first list contains the lower limits of frequency bands and the 
        second list the upper limits.
    fs : int
        Frequency rate.

    Returns
    -------
    Connectivity matrix of the given con_type.

    """
    
    if con_type == 'corr': 
        return dyconnmap.fc.corr(label_ts, fb, fs)
    
    elif con_type == 'coherence': 
        return dyconnmap.fc.coherence(label_ts, fb, fs)
   
    elif con_type == 'pli':
        pli = dyconnmap.fc.pli(label_ts, fb, fs)
        return pli[1]
    
    elif con_type == 'plv':
        plv = dyconnmap.fc.plv(label_ts, fb, fs)
        return plv[1]
    
    elif con_type == 'lps':
        lps1 = lps(label_ts, fb, fs)
        return lps1   
        
    else: 
        raise NameError('con_type "' + con_type + '" is not an option')
        
    
##############################################################################
# def getConnectivity(dir_ts, dir_y_ID, dir_save, con_type, lower, upper):
#     """
#     Calculates and saves the pairwise connectivity measures in one matrix per 
#     subject.
    
#     Parameters
#     ----------
#     dir_ts : string
#         Directory path to where the time series can be found.
#     dir_y_ID : string
#         Directory path to where the y-vector can be extracted.    
#     dir_save : string
#         Directory path to where the results should be saved.
#     con_type : string
#         The desired connectivity measure.
#     lower : list of floats
#         Lower limit of the desired frequency ranges. Needs to have same length
#         as upper.
#     upper : list of floats
#         Upper limit of the desired frequency ranges. Needs to have same length
#         as lower.

#     Notes
#     -------
#     Saves a dictionary with all the connectivity features for each subject, 
#     and two matrices: one with the average connectivity for all schizophrenic
#     patients and one with the healthy controls.

#     """
#     # Initialize values
#     features = {}
#     con_mat_scz = {}
#     count_scz = 0
#     con_mat_hc = {}
#     count_hc = 0
    
#     freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    
#     # Create folder if it does not exist
#     if not op.exists(dir_save):
#         mkdir(dir_save)
#         print('\nCreated new path : ' + dir_save)
    
#     sub_classes = getSubjectClass(dir_y_ID)
    
#     for file in tqdm(listdir(dir_ts)):
#         with open(dir_ts+file, 'rb') as file:
#             label_dict = pickle.load(file)
#         #print(label_ts)
#         label_ts = np.array(list(label_dict['timeseries']))
        
#         subject = str(file).split('Subjects_ts/')[1].split('RSA_')[0]
#         #pdb.set_trace()
        
#         feature_vec = []
            
#         #else:                
#         for i in range(len(upper)):
#             fb=[lower[i], upper[i]]
#             con_mat = connectivity(con_type, label_ts, fb, fs=256)
#             feature_vec.extend(list(con_mat[np.triu_indices(len(con_mat), k=1)]))
            
#             if sub_classes[subject]:
#                 count_scz += 1
#                 try: 
#                     con_mat_scz[freq_bands[i]] += con_mat.copy()
#                 except KeyError:
#                     con_mat_scz[freq_bands[i]] = con_mat.copy()
#             else: 
#                 count_hc += 1                        
#                 try: 
#                     con_mat_hc[freq_bands[i]] += con_mat.copy()
#                 except KeyError:
#                     con_mat_hc[freq_bands[i]] = con_mat.copy()
        
        
#         ###################################################################
#         # Construct dictionary and save 
#         ###################################################################
#         features[subject] = np.array(feature_vec)
#         feature_dict = {'features' : features}
#         # Save to computer 
#         save_name = dir_save + '/feature_dict_' + con_type +'.pkl'
        
#         with open(save_name, 'wb') as file:
#             pickle.dump(feature_dict, file )
            
#     # Save connectivity matrices
#     for band in freq_bands:
#         con_mat_scz[band] = con_mat_scz[band]/count_scz    
#         con_mat_hc[band] = con_mat_hc[band]/count_hc
    
#     save_name = dir_save + '/avg_' + con_type + '_mat_scz.pkl'
#     with open(save_name, 'wb') as file:
#         pickle.dump(con_mat_scz, file)
        
#     save_name = dir_save + '/avg_' + con_type + '_mat_hc.pkl'
#     with open(save_name, 'wb') as file:
#         pickle.dump(con_mat_hc, file)
        
##############################################################################
class RunningStats:
    '''
    Class to calculate mean and standard deviation for the connectivity 
    matrices. Not used yet.
    '''
    def __init__(self,dim):
        self.dim = dim
        self.n = 0
        self.old_m = np.zeros([self.dim, self.dim])
        self.new_m = np.zeros([self.dim, self.dim])
        self.old_s = np.zeros([self.dim, self.dim])
        self.new_s = np.zeros([self.dim, self.dim])
        
    def clear(self):
        self.n = 0 
    
    def push(self, x):
        self.n += 1
        
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros([self.dim, self.dim])
        else:
            self.new_m = self.old_m + (x-self.old_m)/self.n
            self.new_s = self.old_s + (x-self.old_m)*(x-self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s
    
    def mean(self):
        return self.new_m if self.n else np.zeros([self.dim, self.dim])
    
    def variance(self):
        return self.new_s/(self.n-1) if self.n > 1 else np.zeros([self.dim, self.dim])
    
    def std(self):
        return np.sqrt(self.variance())
        
##############################################################################      
def getConnectivity2(dir_ts, dir_y_ID, dir_save, con_type, lower, upper):
    """
    Code that will use the runningstate eventually. Not used yet. 
    
    Parameters
    ----------
    dir_ts : string
        Directory path to where the time series can be found.
    dir_y_ID : string
        Directory path to where the y-vector can be extracted.    
    dir_save : string
        Directory path to where the results should be saved.
    con_type : string
        The desired connectivity measure.
    lower : list of floats
        Lower limit of the desired frequency ranges. Needs to have same length
        as upper.
    upper : list of floats
        Upper limit of the desired frequency ranges. Needs to have same length
        as lower.

    Notes
    -------
    Saves a dictionary with all the connectivity features for each subject, 
    and two matrices: one with the average connectivity for all schizophrenic
    patients and one with the healthy controls. Also save the t-statistics. 

    """
    
    # Initialize values
    features = {}
    con_mat_scz = {}
    count_scz = 0
    con_mat_hc = {}
    count_hc = 0
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    
    # Get the class conrresponding to each subject
    sub_classes = getSubjectClass(dir_y_ID)
    
    for file in tqdm(listdir(dir_ts)):
        with open(dir_ts+file, 'rb') as file:
            label_dict = pickle.load(file)

        # Extract the timeseries from the dictionary
        label_ts = np.array(list(label_dict['timeseries']))
        
        subject = str(file).split('Subjects_ts/')[1].split('RSA_')[0]
        
        feature_vec = []
            
        #else:                
        for i in range(len(upper)):
            fb=[lower[i], upper[i]]
            con_mat = connectivity(con_type, label_ts, fb, fs=256)
            feature_vec.extend(list(con_mat[np.triu_indices(len(con_mat), k=1)]))
            
            # If the subjecti is schizophrenic
            if sub_classes[subject]:
                count_scz += 1
                try: 
                    con_mat_scz[freq_bands[i]].push(con_mat.copy())
                except KeyError: # Used to allocate space
                    con_mat_scz[freq_bands[i]] = RunningStats(con_mat.shape[0])
                    con_mat_scz[freq_bands[i]].push(con_mat.copy())
            # If the subject is schizophrenic
            else: 
                count_hc += 1                        
                try: 
                    con_mat_hc[freq_bands[i]].push(con_mat.copy())
                except KeyError: # Used to allocate space
                    con_mat_hc[freq_bands[i]] = RunningStats(con_mat.shape[0])
                    con_mat_hc[freq_bands[i]].push(con_mat.copy())
        
        
        ###################################################################
        # Construct dictionary and save 
        ###################################################################
        features[subject] = np.array(feature_vec)
        feature_dict = {'features' : features}
        # Save to computer 
        save_name = dir_save + '/feature_dict_' + con_type +'.pkl'
        
        with open(save_name, 'wb') as file:
            pickle.dump(feature_dict, file )
            
    #pdb.set_trace()       
    # Save connectivity matrices
    con_mat_scz_mean = {}
    con_mat_scz_var = {}
    con_mat_hc_mean = {}
    con_mat_hc_var = {}
    t_stat = {}
    for band in freq_bands:
        con_mat_scz_mean[band] = con_mat_scz[band].mean()
        con_mat_scz_var[band]  = con_mat_scz[band].variance()
        con_mat_hc_mean[band]  = con_mat_hc[band].mean()
        con_mat_hc_var[band]   = con_mat_hc[band].variance()
        
        # Make t-statistics
        t_st = (con_mat_scz_mean[band]-con_mat_hc_mean[band])/ np.sqrt(con_mat_scz_var[band]+con_mat_hc_var[band])
        nans = np.isnan(t_st)
        t_st[nans] = 0
        t_stat[band] = t_st
    
    save_name = dir_save + '/avg_' + con_type + '_mat_scz.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_scz_mean, file)
       
    save_name = dir_save + '/var_' + con_type + '_mat_scz.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_scz_var, file)
       
    save_name = dir_save + '/avg_' + con_type + '_mat_hc.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_hc_mean, file)
        
    save_name = dir_save + '/var_' + con_type + '_mat_hc.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_hc_var, file)
    
    save_name = dir_save + '/t_stat_' + con_type + '.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(t_stat, file)
        




