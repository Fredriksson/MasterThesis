#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:47:45 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

import numpy as np
import pickle
import pandas as pd
from glob import glob
from datetime import datetime

import pdb #For debugging add pdb.set_trace() in function use c for continue, u for up, exit for exiting debug mode etc.

#{}
#[]


#%%
##############################################################################
def getNewestFolderDate(dir_folders):
    """
    Gets the folder that is ending with the newest date. The folders date 
    should be of the form %d%m, thus first day and then month with nothing in 
    between.
    
    Parameters
    ----------
    dir_folders : string
        Directory path to the folders you want to find the newest date for. 
        The string should not contain the dates.

    Returns
    -------
    newest_date : string
        The newest folder date, in the form %d%m.

    """
    folders = glob(dir_folders + '*')
    
    newest_date = datetime.strptime('19000101', "%Y%d%m").date()
    for path in folders:
        folder_date = datetime.strptime('2020' + path[-4:], "%Y%d%m").date()
        if folder_date > newest_date:
            newest_date = folder_date
    newest_date = newest_date.strftime("%d%m")

    return newest_date


##############################################################################
def get_Xy(dir_features, dir_y_ID, con_type, partialData = False):
    """
    Parameters
    ----------
    dir_features : string
        Directory path to where the features are saved.
    dir_y_ID : string
        Directory path to where the y-vector can be extracted.
    con_type : string
        The desired connectivity measure.
    partialData : boolean (default False)
        Used to chose wether the six  noisy subjects should be included or not.
        False = the full data set

    Returns
    -------
    X : array of arrays
        Matrix containing a vector with all the features for each subject.
        Dimension (number of subjects)x(number of features).
    y : array
        A vector containing the class-information. 
        Remember: 1 = healty controls, 0 = schizophrenic

    """
   
    # Make directory path and get file    
    file_path = dir_features + '/feature_dict_' + con_type + '.pkl'
    with open(file_path, 'rb') as file:
        feature_dict = pickle.load(file)
    
    # Load csv with y classes
    try:
        dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
        if partialData:
            dat = dat[~dat['Id'].isin(['D950', 'D935', 'D259', 'D255', 'D247', 'D160'])]    
    except: 
        dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Group'], sep = ';')

    
    
    X = []
    y = []
    for i, row in dat.iterrows():
        X.append(feature_dict['features'][row['Id']])
        y.append(row['Group'])
    X = np.array(X)
    y = 1 - pd.Series(y)
    #pdb.set_trace()
    return X, y