#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:56:16 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

import numpy as np
import os.path as op
import pickle
from os import listdir
from datetime import datetime
from tqdm import tqdm #count for loops
import matplotlib.pyplot as plt
from os import mkdir
from runOnce_connectivity import connectivity

import pdb #For debugging add pdb.set_trace() in function use c for continue, u for up, exit for exiting debug mode etc.

#{}
#[]

#%%###########################################################################
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
def plotTstatConnectivity(dir_avg_mat, dir_save, freq_band_type, title):
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
        t_stat_mat= pickle.load(file)
    
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    max_val = np.max([np.max(t_stat_mat[band]) for band in t_stat_mat.keys()])   
    min_val = np.min([np.min(t_stat_mat[band]) for band in t_stat_mat.keys()])   
    fig, ax = plt.subplots(3,2, figsize = (10,12))
    for i in range(len(freq_bands)):
        
        con_fig = ax[i//2][i%2].imshow(t_stat_mat[freq_bands[i]], cmap='viridis', vmin= min_val, vmax= max_val)#, clim= np.percentile(con_mat[5, 95]))

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
    save_name = dir_save + dir_avg_mat.split('FeaturesNew')[-1][0:-4] + '_' + freq_band_type + '.jpg'
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
    #pdb.set_trace()
    fig, ax = plt.subplots(3,2, figsize = (10,12))
    for i in range(len(freq_bands)):
        
        if atlas == 'DKEgill':
            # Old rois with paracentral, used in the report
            # rois = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
            # New rois with parahippocampal
            rois =  np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
            
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