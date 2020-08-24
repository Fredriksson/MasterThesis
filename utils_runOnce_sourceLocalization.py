#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:42:44 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

import mne

import os.path as op
import timeit
import pickle
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne.datasets import fetch_fsaverage
from mne import read_labels_from_annot
from os import listdir
from tqdm import tqdm #count for loops
from os import mkdir

import pdb #For debugging add pdb.set_trace() in function use c for continue, u for up, exit for exiting debug mode etc.

# []
# {}

#%%
##############################################################################
def extract_ts(dir_prepro_dat, dir_save, lower, upper, atlas):  
    """
    Parameters
    ----------
    dir_prepro_dat : string
        Path to saved preprocessed data.
    dir_save : string
        Path to where the extracted time series should be saved.
    lower : list of floats
        Lower limit of the desired frequency ranges. Needs to have same length
        as upper.
    upper : list of floats
        Upper limit of the desired frequency ranges. Needs to have same length
        as lower.

    Notes
    ----------
    Saves the extracted time series and the run time as a dictionary, in the 
    chosen path (dir_save).

    """    

    ##################################################################
    # Initialize parameters
    ##################################################################

    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir) 
    #fs_dir = '/home/kmsa/mne_data/MNE-fsaverage-data/fsaverage' 
    #subjects_dir = '/home/kmsa/mne_data/MNE-fsaverage-data'
        
    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = mne.setup_source_space(subject, spacing='oct6',
                                 subjects_dir=None, add_dist=False)
    # Boundary element method
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    
    # Inverse parameters
    method = "eLORETA" #other options are minimum norm, dSPM, and sLORETA
    snr = 3.
    lambda2 = 1. / snr ** 2
    buff_sz = 250
    
    
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    
    # Check what atlas to use and read labels
    if atlas == 'DK':
        # Desikan-Killiany Atlas = aparc
        parc = 'Yeo2011_7Networks_N1000' # 'aparc'
        labels = read_labels_from_annot(subject, parc = 'aparc', hemi='both',
                                     surf_name= 'white', annot_fname = None, regexp = None,
                                     subjects_dir = subjects_dir, verbose = None)
        
        labels = labels[:-1]
    # elif atlas == 'BA':
    #     # Broadmann areas
    #     labels = read_labels_from_annot(subject, parc = 'PALS_B12_Brodmann', hemi='both',
    #                                  surf_name= 'white', annot_fname = None, regexp = None,
    #                                  subjects_dir = subjects_dir, verbose = None)
    #     labels = labels[5:87]
        
    elif atlas == 'BAita':
        # Brodmann areas collected as in Di Lorenzo et al.
        labels = read_labels_from_annot(subject, parc = 'PALS_B12_Brodmann', hemi='both',
                                     surf_name= 'white', annot_fname = None, regexp = None,
                                     subjects_dir = subjects_dir, verbose = None)
        labels = labels[5:87]
        lab_dict = {}
        for lab in labels:
            lab_dict[lab.name] = lab
            
        
        ita_ba = [[1,2,3,4], [5,7], [6,8], [9,10], [11,47], [44,45,46], #[13],
               [20,21,22,38,41,42], [24,25,32], #[24,25,32,33], 
               [23,29,30,31],
               [27,28,35,36], #[27,28,34,35,36], 
               [39,40,43], [19,37], [17,18]]
        # ita_label = ['SMA', 'SPL', 'SFC', 'AFC', 'OFC', 'LFC', #'INS',
        #              'LTL', 'ACC_new', 'PCC', 'PHG_new', 'IPL', 'FLC', 'PVC']
        
        # Sort labels according to connectivity featurers
        new_label = []
        for idx, i in enumerate(ita_ba):
            for j in i:
                ba_lh = 'Brodmann.' + str(j) +'-lh'
                ba_rh = 'Brodmann.' + str(j) +'-rh'  
                
                if j == i[0]:
                    sum_lh = lab_dict[ba_lh]
                    sum_rh = lab_dict[ba_rh]
                else:
                    sum_lh += lab_dict[ba_lh]
                    sum_rh += lab_dict[ba_rh]
            new_label.append(sum_lh)
            new_label.append(sum_rh)
            
        labels = new_label 
        
    elif atlas == 'DKLobes':
        # Brodmann areas collected as in Di Lorenzo et al.
        labels = read_labels_from_annot(subject, parc = 'aparc', hemi='both',
                                     surf_name= 'white', annot_fname = None, regexp = None,
                                     subjects_dir = subjects_dir, verbose = None)
        # Divide into lobes based on 
        # https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
        frontal = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 
                   'parsopercularis', 'parstriangularis', 'parsorbitalis', 
                   'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 
                   'paracentral', 'frontalpole', 'rostralanteriorcingulate',
                   'caudalanteriorcingulate']
        parietal = ['superiorparietal', 'inferiorparietal', 'supramarginal', 
                    'postcentral', 'precuneus', 'posteriorcingulate', 
                    'isthmuscingulate']
        temporal = ['superiortemporal', 'middletemporal', 'inferiortemporal',
                    'bankssts', 'fusiform', 'transversetemporal', 'entorhinal', 
                    'temporalpole', 'parahippocampal']        
        occipital = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']
        
        all_lobes = {'frontal': frontal, 'parietal': parietal, 'occipital': occipital, 'temporal': temporal}
        
        labels = labels[:-1]
        lab_dict = {}
        for lab in labels:
            lab_dict[lab.name] = lab
        
        # Sort labels according to connectivity featurers
        new_label = []
        for lobes in list(all_lobes.keys()):
            for idx, name in enumerate(all_lobes[lobes]):
                name_lh = name +'-lh'
                name_rh = name +'-rh'  
                
                if idx == 0:
                    sum_lh = lab_dict[name_lh]
                    sum_rh = lab_dict[name_rh]
                else:
                    sum_lh += lab_dict[name_lh]
                    sum_rh += lab_dict[name_rh]
            sum_lh.name = lobes + '-lh'
            sum_rh.name = lobes + '-rh'
            new_label.append(sum_lh)
            new_label.append(sum_rh)
            
        labels = new_label 
    # elif finished    
        
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
        
    # List of time series that have already been saved
    already_saved = [i.split('_' + atlas)[0] for i in listdir(dir_save)]

    count = 0
    run_time = 0
    
    for filename in tqdm(listdir(dir_prepro_dat)):
        # Only loop over ".set" files
        if not filename.endswith(".set"):
            continue
        
        # Only choose the files that are not already in the save directory
        if filename.split('.')[0] in already_saved:
            count += 1
            continue
        
        start = timeit.default_timer()
        timeseries_dict = {}
        ###################################
        # Load preprocessed data
        ###################################
        ID_list = op.join(dir_prepro_dat,filename)
        raw = mne.io.read_raw_eeglab(ID_list,preload=True);
       
        # Set montage (number of used channels = 64)
        raw.set_montage('biosemi64')            
    
        ##################################################################
        # Forward soultion: from brain to electrode
        ##################################################################
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                        bem=bem, eeg=True, mindist=5.0, n_jobs=-1);
        
        ##################################################################
        # Inverse modeling
        ##################################################################
        # Compute noise covariance
        noise_cov = mne.compute_raw_covariance(raw, n_jobs=-1);  
        # make an EEG inverse operator
        inverse_operator = make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8);
        
        raw.set_eeg_reference('average', projection=True)
        
        # Hardcoded print to give an overview of how much is done and left
        print('\n####################################################'+ 
              '\n####################################################'+
              '\n####################################################'+
              '\nSubject number: ' + str(count) +', '+ filename.split('.')[0] + 
              '\nRun time/subject = ' + str(run_time)+
              '\nRun time left in hours ~' + str((85-(count+1))*(run_time/60)) +
              '\n####################################################'+ 
              '\n####################################################'+
              '\n####################################################')
        
        # Compute inverse solution
        stc = apply_inverse_raw(raw, inverse_operator, lambda2,
                                      method=method, nave=1, pick_ori=None,
                                      verbose=True, buffer_size=buff_sz); 
        #pdb.set_trace()
        del raw
        
        # Hardcoded print to give an overview of how much is done and left
        print('\n####################################################'+ 
              '\n####################################################'+
              '\n####################################################'+
              '\nSubject number: ' + str(count) +', '+ filename.split('.')[0] + 
              '\nRun time/subject = ' + str(run_time)+
              '\nRun time left in hours ~' + str((85-(count+1))*(run_time/60)) +
              '\n####################################################'+ 
              '\n####################################################'+
              '\n####################################################')
        ##################################################################
        # Extract timeseries from DK regions
        ##################################################################
        # Label time series by Desikan-Killiany Atlas -> 68 ts
        label_ts = mne.extract_label_time_course(stc, labels, inverse_operator['src'], 
                                                 mode= 'pca_flip', return_generator=True);
        del stc
        ###################################################################
        # Construct and save dictionary
        ###################################################################
        subject = filename.split('_')[0]        
        stop = timeit.default_timer()
        run_time = (stop - start)/60
        timeseries_dict = {'timeseries' : label_ts, 'time' : run_time}
        del label_ts
        # Save to computer 
        save_name = dir_save + '/' + subject + '_' + atlas + '_timeseries' +'.pkl'
        #save_name = '/share/FannyMaster/PythonNew/DK_timeseries/DK_source_timeseries_'+ date +'.pkl'
        
        with open(save_name, 'wb') as file:
            pickle.dump(timeseries_dict, file )
        
        del timeseries_dict
        
        count += 1
    
##############################################################################



    