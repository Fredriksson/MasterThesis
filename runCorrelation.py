"""
Created on Wed Mar 25 13:58:58 2020

@author: Fanny 
"""

import numpy as np
import mne
import os.path as op
import dyconnmap
import timeit
import pickle

#from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, compute_proj_ecg, compute_proj_eog, corrmap, find_eog_events)
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne.datasets import fetch_fsaverage
from mne import read_labels_from_annot
#from mne.connectivity import spectral_connectivity, envelope_correlation
from os import listdir
from datetime import datetime
from tqdm import tqdm #count for loops

#%%
features = {}
feature_dict = {}
info = {}
loop_time = []


#%%##################################################################
# Initialize parameters
##################################################################
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
    
# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)
# Boundary element method
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# Inverse parameters
method = "eLORETA" #other options are minimum norm, dSPM, and sLORETA
snr = 3.
lambda2 = 1. / snr ** 2
buff_sz = 250

# Desikan-Killiany Atlas 
labels = read_labels_from_annot(subject, parc = 'aparc', hemi='both',
                                 surf_name= 'white', annot_fname = None, regexp = None,
                                 subjects_dir = None, verbose = None)

# Frequency bands 
lower = [1.5, 4.,  8., 12., 20., 30.]
upper = [4. , 8., 12., 20., 30., 80.]

# Date
date = datetime.now().strftime("%d%m")

#%%
#directory = r'C:\Users\fanny\Documents\Thesis\Python'
directory =  r'P:\PC Glostrup\Lukkede Mapper\Forskningsenhed\CNSR\Fanny\Preprocessed'
count = 0;
for filename in tqdm(listdir(directory)):
    # Only loop over ".set" files
    if not filename.endswith(".set"):
        continue
    start = timeit.default_timer()

    ###################################
    # Load preprocessed data
    ###################################
    ID_list = op.join(directory,filename)
    raw = mne.io.read_raw_eeglab(ID_list,preload=True)
    count += 1; 
    print(count)
    
    # Set montage (number of used channels = 64)
    raw.set_montage('biosemi64')
    # Define channels
    eeg_channels = mne.pick_channels(raw.info['ch_names'],include=[],exclude=[])
    # Sampling frequency
    sfreq = raw.info['sfreq']  # the sampling frequency
        

    ##################################################################
    # Forward soultion: from brain to electrode
    ##################################################################
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
#%%    
    ##################################################################
    # Inverse modeling
    ##################################################################
    # Compute noise covariance
    noise_cov = mne.compute_raw_covariance(raw)  
    # make an EEG inverse operator
    inverse_operator = make_inverse_operator(
        raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
    
    raw.set_eeg_reference('average', projection=True)

    # Compute inverse solution
    stc = apply_inverse_raw(raw, inverse_operator, lambda2,
                                  method=method, nave=1, pick_ori=None,
                                  verbose=True, buffer_size=buff_sz) 
    
    ##################################################################
    # Find connectivity matrix
    ##################################################################
    # Label time series by Desikan-Killiany Atlas -> 68 ts
    label_ts = mne.extract_label_time_course(stc, labels[:-1], inverse_operator['src'], 
                                             mode= 'pca_flip', return_generator=True)
    
    # Calculate correlation    
    feature_vec = []
    
    for i in range(len(upper)):
        corr_mat = dyconnmap.fc.corr(label_ts, fb=[lower[i], upper[i]], fs= 256)
        feature_vec.extend(list(corr_mat[np.triu_indices(len(corr_mat), k=1)]))

    ###################################################################
    # Construct and save dictionary
    ###################################################################
    subject = filename.split('.')[0]
    features[subject] = np.array(feature_vec)
    stop = timeit.default_timer()
    info[subject] = {'time' : (stop - start)/60}
    feature_dict = {'features' : features, 'info' : info}
    
    # Save to computer 
    save_name = 'Features/feature_dict_'+ date +'.pkl'
    
    with open(save_name, 'wb') as file:
        pickle.dump(feature_dict, file )
    
    # Delete values 
    del raw, stc, label_ts
   

#%%
tot_time = sum([info_vals['time'] for info_vals in list(feature_dict['info'].values())])
print(f'Total Time: {tot_time:.2f}')
print(f'Average Time: {tot_time/len(feature_dict["features"].keys()):.2f}')    
    
#with open('Features/feature_dict.pkl', 'rb') as file:
#    test = pickle.load(file )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##############################################################################