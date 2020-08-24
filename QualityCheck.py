#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:19:24 2020

@author: kmsa
"""
#%% Check quality of the pre-processed data from EEGlab

import mne
import os.path as op
from os import listdir
from tqdm import tqdm #count for loops

#%%
directory =  r'/share/FannyMaster/Preprocessed'

subjects = sorted(listdir(directory)) #list of all files in directory
subjects = [i for i in subjects if i.endswith('.set')] #list of .set-files

count = 0#First subject to check, 0-indexed
subjects = subjects[count:]
for filename in tqdm(subjects):
    print(filename)
    # Load preprocessed data
    ID_list = op.join(directory,filename)
    raw = mne.io.read_raw_eeglab(ID_list,preload=False)
    count += 1; 
    print(count)
    
    # Plot data
    for i in range(0,600,20):
        mne.viz.plot_raw(raw,duration=20,start=i,n_channels=64)
        # Pause until data is checked
        input("Press enter to continue...")

    # Pause until data is checked
    input("Press enter to continue...")
    
    