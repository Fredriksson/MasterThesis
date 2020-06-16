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
from datetime import datetime
from tqdm import tqdm #count for loops
import matplotlib.pyplot as plt
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
def getConnectivity(dir_ts, dir_y_ID, dir_save, con_type, lower, upper):
    """
    Calculates and saves the pairwise connectivity measures in one matrix per 
    subject.
    
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
    patients and one with the healthy controls.

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
    
    sub_classes = getSubjectClass(dir_y_ID)
    
    for file in tqdm(listdir(dir_ts)):
        with open(dir_ts+file, 'rb') as file:
            label_dict = pickle.load(file)
        #print(label_ts)
        label_ts = np.array(list(label_dict['timeseries']))
        
        subject = str(file).split('Subjects_ts/')[1].split('RSA_')[0]
        #pdb.set_trace()
        
        feature_vec = []
            
        #else:                
        for i in range(len(upper)):
            fb=[lower[i], upper[i]]
            con_mat = connectivity(con_type, label_ts, fb, fs=256)
            feature_vec.extend(list(con_mat[np.triu_indices(len(con_mat), k=1)]))
            
            if sub_classes[subject]:
                count_scz += 1
                try: 
                    con_mat_scz[freq_bands[i]] += con_mat.copy()
                except KeyError:
                    con_mat_scz[freq_bands[i]] = con_mat.copy()
            else: 
                count_hc += 1                        
                try: 
                    con_mat_hc[freq_bands[i]] += con_mat.copy()
                except KeyError:
                    con_mat_hc[freq_bands[i]] = con_mat.copy()
        
        
        ###################################################################
        # Construct dictionary and save 
        ###################################################################
        features[subject] = np.array(feature_vec)
        feature_dict = {'features' : features}
        # Save to computer 
        save_name = dir_save + '/feature_dict_' + con_type +'.pkl'
        
        with open(save_name, 'wb') as file:
            pickle.dump(feature_dict, file )
            
    # Save connectivity matrices
    for band in freq_bands:
        con_mat_scz[band] = con_mat_scz[band]/count_scz    
        con_mat_hc[band] = con_mat_hc[band]/count_hc
    
    save_name = dir_save + '/avg_' + con_type + '_mat_scz.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_scz, file)
        
    save_name = dir_save + '/avg_' + con_type + '_mat_hc.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_hc, file)
        
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
    patients and one with the healthy controls. Should be saving standard 
    deviation later on as well. 

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
            
    # Save connectivity matrices
    for band in freq_bands:
        con_mat_scz[band] = con_mat_scz[band]/count_scz    
        con_mat_hc[band] = con_mat_hc[band]/count_hc
    
    save_name = dir_save + '/avg_' + con_type + '_mat_scz.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_scz, file)
        
    save_name = dir_save + '/avg_' + con_type + '_mat_hc.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(con_mat_hc, file)
    
##############################################################################    
def plotConnectivity(dir_ts, dir_save, con_type, lower, upper, sub_nb):
    """
    Plots the connectivity matrix for one given subject as a test.
    
    Parameters
    ----------
    dir_ts : string
        Directory path to where the time series can be found.
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
    sub_nb : int
        Subject number to use.

    Notes
    -------
    Saves a figure with the connectivity matrices for each band for subject 
    number sub_nb.

    """
    
    file = listdir(dir_ts)[sub_nb]
    with open(dir_ts+file, 'rb') as file:
        label_dict = pickle.load(file)
    label_ts = np.array(list(label_dict['timeseries']))
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    fig, ax = plt.subplots(2,3)
    for i in tqdm(range(len(upper))):
        fb=[lower[i], upper[i]]
        con_mat = connectivity(con_type, label_ts, fb, fs=256)
        
        #con_fig = ax[0][0].imshow(con_mat, cmap='viridis')#, clim= np.percentile(con_mat[5, 95]))
        con_fig = ax[i//3][i%3].imshow(con_mat, cmap='viridis', vmin= -1, vmax= 1)#, clim= np.percentile(con_mat[5, 95]))
        #ax[0][0].tight_layout()
        #plt.colorbar(con_fig)
        ax[i//3][i%3].set_title(freq_bands[i])
    
    
    plt.tight_layout(pad = 0.6)
    fig.suptitle(con_type.capitalize())
    fig.subplots_adjust(top=0.88)
    fig.colorbar(con_fig, ax = ax.ravel().tolist(), shrink = 0.98, ticks = [-1, 0, 1])
    plt.show()
    
    # Save to computer 
    date = datetime.now().strftime("%d%m")
    save_name = dir_save + '/Connectivity_' + con_type + '_' + date +'.jpg'
    fig.savefig(save_name, bbox_inches = 'tight')
            
##############################################################################
def plotAvgConnectivity(dir_avg_mat, dir_save, freq_band_type, title):
    """
    Plots the saved average connectivity matrices for all the bands.
    
    Parameters
    ----------
    dir_avg_mat : string
        Directory path to where the averge connectivity matrix can be found.
    dir_save : string
        Directory path to where the results should be saved.
    freq_band_type : string
        What Frequency ranges that are used.
    title : string
        Title to print on the plot.

    Notes
    -------
    Saves a figure with the average connectivity matrices for each band.

    """    
    
    with open(dir_avg_mat, 'rb') as file:
        avg_mat= pickle.load(file)
    
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    fig, ax = plt.subplots(3,2, figsize = (10,12))
    for i in range(len(freq_bands)):
        
        con_fig = ax[i//2][i%2].imshow(avg_mat[freq_bands[i]], cmap='viridis') #, vmin= 0, vmax= 1)#, clim= np.percentile(con_mat[5, 95]))

        ax[i//2][i%2].set_title(freq_bands[i], fontsize=30)
        ax[i//2][i%2].tick_params(labelsize=25)
        ax[i//2][i%2].set_xlabel('Brain Area', fontsize = 25)
        ax[i//2][i%2].set_ylabel('Brain Area', fontsize = 25)
    
    
    plt.tight_layout(pad = 0.6)
    fig.suptitle(title, fontsize = 33)
    fig.subplots_adjust(top=0.9, hspace = 0.55,  bottom = 0.05)

    cbar = fig.colorbar(con_fig, ax = ax.ravel().tolist())#, ticks = [0, 0.5, 1])
    cbar.ax.tick_params(labelsize=25)
    plt.show()
    
    # Save to computer 
    save_name = dir_save + dir_avg_mat.split('Features')[-1][0:-4] + '_' + freq_band_type + '.jpg'
    fig.savefig(save_name, bbox_inches = 'tight')


##############################################################################
def plotAvgConnectivity2(dir_avg_mat1, dir_avg_mat2, dir_save, freq_band_type, title, atlas):
    """
    Plots the saved average connectivity matrices for all the bands for both 
    healthy controls and schizophrenic subjects.
    
    Parameters
    ----------
    dir_avg_mat1 : string
        Directory path to where one type of averge connectivity matrices.
    dir_avg_mat2 : string
        Directory path to where the other type of averge connectivity matrices.
    dir_save : string
        Directory path to where the results should be saved.
    freq_band_type : string
        What Frequency ranges that are used.
    title : string
        Title to print on the plot.
    atlas : 
        What atlas that have been used to calculate the average matrices.

    Notes
    -------
    Saves two figure: One for the first type of average matrices, and one for
    the others.

    """    
    
    with open(dir_avg_mat1, 'rb') as file:
        avg_mat1 = pickle.load(file)
    
    with open(dir_avg_mat2, 'rb') as file:
        avg_mat2 = pickle.load(file)
    
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
        
    # Used to make the plots having the same range
    max_val1 = np.max([np.max(avg_mat1[band]) for band in avg_mat1.keys()])    
    max_val = np.max([np.max(avg_mat2[band]) for band in avg_mat2.keys()])    
    if max_val1>max_val:
        max_val = max_val1
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    fig, ax = plt.subplots(3,2, figsize = (10,12))
    for i in range(len(freq_bands)):
        
        if atlas == 'DKEgill':
            # Old rois with paracentral, used in the report
            rois = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
            # New rois with parahippocampal
            # rois =  np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
            
            rois = np.array([rois,rois]).reshape(-1,order='F')

            avg_mat1[freq_bands[i]] = avg_mat1[freq_bands[i]][np.nonzero(rois)][:,np.nonzero(rois)].squeeze()
            avg_mat2[freq_bands[i]] = avg_mat2[freq_bands[i]][np.nonzero(rois)][:,np.nonzero(rois)].squeeze()
        
        
        con_fig = ax[i//2][i%2].imshow(avg_mat1[freq_bands[i]], cmap='viridis', vmin= 0, vmax= max_val)#, clim= np.percentile(con_mat[5, 95]))

        ax[i//2][i%2].set_title(freq_bands[i], fontsize=30)
        ax[i//2][i%2].tick_params(labelsize=25)
        ax[i//2][i%2].set_xlabel('Brain Area', fontsize = 25)
        ax[i//2][i%2].set_ylabel('Brain Area', fontsize = 25)
    
    
    plt.tight_layout(pad = 0.6)
    fig.suptitle(title + 'hc', fontsize = 33)
    fig.subplots_adjust(top=0.9, hspace = 0.55,  bottom = 0.05, wspace = -0.3)

    plt.show()
    
    save_name = dir_save + dir_avg_mat1.split('Features')[-1][0:-4] + '_' + freq_band_type + '.jpg'
    fig.savefig(save_name, bbox_inches = 'tight')
            
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    fig, ax = plt.subplots(3,2, figsize = (10,12))
    for i in range(len(freq_bands)):
        
        con_fig = ax[i//2][i%2].imshow(avg_mat2[freq_bands[i]], cmap='viridis', vmin= 0, vmax= max_val)#, clim= np.percentile(con_mat[5, 95]))
  
        ax[i//2][i%2].set_title(freq_bands[i], fontsize=30)
        ax[i//2][i%2].tick_params(labelsize=25)
        ax[i//2][i%2].set_xlabel('Brain Area', fontsize = 25)
        ax[i//2][i%2].set_ylabel('Brain Area', fontsize = 25)
    
    
    plt.tight_layout(pad = 0.6)
    fig.suptitle(title + 'scz', fontsize = 33)
    fig.subplots_adjust(top=0.9, hspace = 0.55,  bottom = 0.05)

    cbar = fig.colorbar(con_fig, ax = ax.ravel().tolist())#, ticks = [0, 0.5, 1])
    cbar.ax.tick_params(labelsize=25)
    plt.show()

    save_name = dir_save + dir_avg_mat2.split('Features')[-1][0:-4] + '_' + freq_band_type + '.jpg'
    fig.savefig(save_name, bbox_inches = 'tight')
##############################################################################








