#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:56:16 2020

@author: s153968
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


#{}
#[]

#%%

def getSubjectClass(dir_y_ID):
    # Load csv with y classes
    sub_classes = {}
    age_gender_dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
    for i in range(len(age_gender_dat)):
        sub_classes[age_gender_dat['Id'][i]] = age_gender_dat['Group'][i]
    
    return sub_classes

##############################################################################
def lps(data, fb, fs):
    _, u_phase, _ = analytic_signal(data, fb, fs) 
    n_channels, _ = np.shape(data)
    pairs = [(r2, r1) for r1 in range(n_channels) for r2 in range(r1)]
    avg = np.zeros((n_channels, n_channels)) 
    for pair in pairs:
        u1, u2 = u_phase[pair,]
        ts_plv = np.exp(1j * (u1-u2))
        
        #avg_plv = np.abs(np.sum(ts_plv)) / float(label_ts.shape[1])
        r = np.sum(ts_plv) / float(data.shape[1])
        num = np.power(np.imag(r), 2)
        denom = 1-np.power(np.real(r), 2)
        #pdb.set_trace()
        avg[pair] = num/denom
    return avg

##############################################################################
#import mlab_Fanny as mlab
# def lps_csd(data, lower, upper, fs, pairs=None):
    
#     # pairs = None
#     # fs = 256
#     # data = label_ts
#     n_channels, _ = np.shape(data)
#    # _, _, filtered = analytic_signal(data, fb, fs)
    
#     if pairs is None:
#         pairs = [(r2, r1) for r1 in range(n_channels) for r2 in range(r1)]
    
#     freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
#     lps = {'delta': np.zeros((n_channels, n_channels)), 
#            'theta': np.zeros((n_channels, n_channels)), 
#            'alpha': np.zeros((n_channels, n_channels)), 
#            'beta1': np.zeros((n_channels, n_channels)), 
#            'beta2': np.zeros((n_channels, n_channels)), 
#            'gamma': np.zeros((n_channels, n_channels))} #np.zeros((n_channels, n_channels))
    
#     for pair in pairs:
        
#         filt1, filt2 = data[pair,]
       
#         csdxy, freqxy = mlab.csd(
#              x=filt1, y=filt2, Fs=fs, scale_by_freq=True, sides="onesided")
         
#         r = csdxy
         
#         num = np.power(np.imag(r),2)
#         denom = 1 - np.power(np.real(r),2)
#         lps_all = num / denom
        
#         for fb in range(len(lower)):
#             lower_lps = 0
#             upper_lps = 0
#             if not float(int(lower[fb]))==lower[fb]:
#                 lower_lps = float(lps_all[int(np.floor(lower[fb])):int(np.ceil(lower[fb]))])*(np.ceil(lower[fb])-lower[fb])
#             if not float(int(upper[fb]))==upper[fb]:
#                upper_lps = float(lps_all[int(np.floor(upper[fb])):int(np.ceil(upper[fb]))])*(np.abs(np.floor(upper[fb])-upper[fb]))
#             lps[freq_bands[fb]][pair] = np.sum(lps_all[int(np.ceil(lower[fb])):int(np.floor(upper[fb]))]) + lower_lps + upper_lps
#     return lps  

##############################################################################
def connectivity(con_type, label_ts, fb, fs):
    """
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
    #print(con_type)
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
    Saves a dictionary with all the connectivity features for each subject, in
    the dir_save path.

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
    Saves a dictionary with all the connectivity features for each subject, in
    the dir_save path.

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
                    con_mat_scz[freq_bands[i]].push(con_mat.copy())
                except KeyError:
                    con_mat_scz[freq_bands[i]] = RunningStats(con_mat.shape[0])
                    con_mat_scz[freq_bands[i]].push(con_mat.copy())
            else: 
                count_hc += 1                        
                try: 
                    con_mat_hc[freq_bands[i]].push(con_mat.copy())
                except KeyError:
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
    Parameters
    ----------
    dir_avg_mat : string
        Directory path to where the averge connectivity matrix can be found.
    dir_save : string
        Directory path to where the results should be saved.
    sub_nb : int
        Subject number to use.

    Notes
    -------
    Saves a figure with the connectivity matrices for each band for subject 
    number sub_nb.

    """    
    
    with open(dir_avg_mat, 'rb') as file:
        avg_mat= pickle.load(file)
    #label_ts = np.array(list(label_dict['timeseries']))
    
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    
    
    
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    fig, ax = plt.subplots(3,2, figsize = (10,12))
    for i in range(len(freq_bands)):
        
        con_fig = ax[i//2][i%2].imshow(avg_mat[freq_bands[i]], cmap='viridis') #, vmin= 0, vmax= 1)#, clim= np.percentile(con_mat[5, 95]))
        #ax[0][0].tight_layout()
        #plt.colorbar(con_fig)
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
    
    
    #pdb.set_trace()
    # Save to computer 
    #date = datetime.now().strftime("%d%m")
    #pdb.set_trace()
    save_name = dir_save + dir_avg_mat.split('Features')[-1][0:-4] + '_' + freq_band_type + '.jpg'
    fig.savefig(save_name, bbox_inches = 'tight')
            
##############################################################################


# freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
#     fig, ax = plt.subplots(2,3, figsize = (12,8))
#     for i in range(len(freq_bands)):
        
#         con_fig = ax[i//3][i%3].imshow(avg_mat[freq_bands[i]], cmap='viridis') #, vmin= 0, vmax= 1)#, clim= np.percentile(con_mat[5, 95]))
#         #ax[0][0].tight_layout()
#         #plt.colorbar(con_fig)
#         ax[i//3][i%3].set_title(freq_bands[i], fontsize=20)
#         ax[i//3][i%3].tick_params(labelsize=20)
#         ax[i//3][i%3].set_xlabel('Brain Area', fontsize = 18)
#         ax[i//3][i%3].set_ylabel('Brain Area', fontsize = 18)
    
    
#     plt.tight_layout(pad = 0.6)
#     fig.suptitle(title, fontsize = 30)
#     fig.subplots_adjust(top=0.9, hspace = 0.3, wspace= 0.4, bottom = 0.12)

#     cbar = fig.colorbar(con_fig, ax = ax.ravel().tolist(), shrink = 0.94)#, ticks = [0, 0.5, 1])
#     cbar.ax.tick_params(labelsize=20)
#     plt.show()




##############################################################################
def plotAvgConnectivity2(dir_avg_mat1, dir_avg_mat2, dir_save, freq_band_type, title, atlas):
    """
    Parameters
    ----------
    dir_avg_mat : string
        Directory path to where the averge connectivity matrix can be found.
    dir_save : string
        Directory path to where the results should be saved.
    sub_nb : int
        Subject number to use.

    Notes
    -------
    Saves a figure with the connectivity matrices for each band for subject 
    number sub_nb.

    """    
    
    with open(dir_avg_mat1, 'rb') as file:
        avg_mat1 = pickle.load(file)
    
    with open(dir_avg_mat2, 'rb') as file:
        avg_mat2 = pickle.load(file)
    
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    
    
    
    max_val1 = np.max([np.max(avg_mat1[band]) for band in avg_mat1.keys()])
    
    max_val = np.max([np.max(avg_mat2[band]) for band in avg_mat2.keys()])
    
    if max_val1>max_val:
        max_val = max_val1
    
    #pdb.set_trace()
    
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
        #ax[0][0].tight_layout()
        #plt.colorbar(con_fig)
        ax[i//2][i%2].set_title(freq_bands[i], fontsize=30)
        ax[i//2][i%2].tick_params(labelsize=25)
        ax[i//2][i%2].set_xlabel('Brain Area', fontsize = 25)
        ax[i//2][i%2].set_ylabel('Brain Area', fontsize = 25)
    
    
    plt.tight_layout(pad = 0.6)
    fig.suptitle(title + 'hc', fontsize = 33)
    fig.subplots_adjust(top=0.9, hspace = 0.55,  bottom = 0.05, wspace = -0.3)

    # cbar = fig.colorbar(con_fig, ax = ax.ravel().tolist())#, ticks = [0, 0.5, 1])
    # cbar.ax.tick_params(labelsize=25)
    plt.show()
    
    save_name = dir_save + dir_avg_mat1.split('Features')[-1][0:-4] + '_' + freq_band_type + '.jpg'
    fig.savefig(save_name, bbox_inches = 'tight')
            
    
    freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    fig, ax = plt.subplots(3,2, figsize = (10,12))
    for i in range(len(freq_bands)):
        
        con_fig = ax[i//2][i%2].imshow(avg_mat2[freq_bands[i]], cmap='viridis', vmin= 0, vmax= max_val)#, clim= np.percentile(con_mat[5, 95]))
        #ax[0][0].tight_layout()
        #plt.colorbar(con_fig)
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
    #pdb.set_trace()
    save_name = dir_save + dir_avg_mat2.split('Features')[-1][0:-4] + '_' + freq_band_type + '.jpg'
    fig.savefig(save_name, bbox_inches = 'tight')
##############################################################################








