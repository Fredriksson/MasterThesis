#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:47:45 2020

@author: s153968
"""

import numpy as np
import os.path as op
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm #count ffor loops
import math
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import preprocessing
from datetime import datetime
from sklearn.utils import shuffle
from os import mkdir
import pdb
import timeit

#{}
#[]

#%%
##############################################################################
# def getDataOld(dir_features, dir_y_ID, con_type, partialData = False):
#     """
#     Parameters
#     ----------
#     dir_features : string
#         Directory path to where the features are saved.
#     dir_y_ID : string
#         Directory path to where the y-vector can be extracted.
#     con_type : string
#         The desired connectivity measure.

#     Returns
#     -------
#     X : array of arrays
#         Matrix containing a vector with all the features for each subject.
#         Dimension (number of subjects)x(number of features).
#     y : array
#         A vector containing the class-information. 
#         Remember: 0 = healty controls, 1 = schizophrenic

#     """
   
#     # Make directory path and get file    
#     file_path = dir_features + '/feature_dict_' + con_type + '.pkl'
#     with open(file_path, 'rb') as file:
#         feature_dict = pickle.load(file)
    
#     # Get X matrix
#     X = np.array(list(feature_dict['features'].values()))
#     #run_time = sum([info_vals['time'] for info_vals in list(feature_dict['info'].values())])
    
#     # Load csv with y classes
#     age_gender_dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
#     #age_gender_dat = age_gender_dat[~age_gender_dat['Id'].isin(['D950', 'D935', 'D259', 'D255', 'D247', 'D160'])]
    
#     # Order y so it matches the rows of X
#     ID = pd.DataFrame(list(feature_dict['features']), columns = ['Id'])
#     ID['Id'] = [i.split('R')[0] for i in ID['Id']]
#     age_gender_dat = ID.merge(age_gender_dat, left_on = 'Id', right_on = 'Id', how = 'outer')
#     y = age_gender_dat['Group'] 
#     y = 1 - y
    
#     return X, y


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
    #pdb.set_trace()
    return X, y


##############################################################################
def save_values(auc, nz_coef_idx, nz_coef_val, clf_name, perms, dir_save, date):
    """
    Parameters
    ----------
    auc : dictionary
        Dictionary with the auc values.
    nz_coef_idx : dictionary
        Dictionary with the non-zero coefficient indices.
    nz_coef_val : dictionary
        Dictionary with the non-zero coefficient values.
    clf_name : string
        String with the name of a classifier.
    perms : range(*)
        Range with desired number (*) of permutations. 
        *=1 indicates no permutations.
    dir_save : string
        Directory path to where the results should be saved.
    date : string
        The date in day and month when the CV-classifier started
        
    Notes
    -------
    Saves three different values in the dir_save path: 
    auc : dictionary
        Contains the auc-scores for each loop, either divided into bands or 
        with the key "all".
    nz_coef_idx : dictionary
        Contains the non-zero coefficient indices for each loop, either 
        divided into bands or with the key "all".
    nz_coef_val : dictionary
        Contains the non-zero coefficient values (the weights) for each 
        loop, either divided into bands or with the key "all".

    """
    # Create folder if it does not exist
    if not op.exists(dir_save):
        mkdir(dir_save)
        print('\nCreated new path : ' + dir_save)
    
    # Check if seperate bands were used or not
    if len(auc.keys()) == 1:
        band_type = 'all'
    else: 
        band_type = 'bands'
    
    name = clf_name +'_'+ band_type + '_' + date
    
    # Check if permutations were used
    if len(perms)>1: 
        perm = '/perm_'
    else: # No permutation
        perm = '/'
        
        save_name = dir_save+perm+'nz_coef_idx_'+name+'.pkl'
        with open(save_name, 'wb') as file:
            pickle.dump(nz_coef_idx, file)
            
        # Save non zero coefficient values
        save_name = dir_save+perm+'nz_coef_val_'+name+'.pkl'
        with open(save_name, 'wb') as file:
            pickle.dump(nz_coef_val, file)
    
    # Save auc
    save_name = dir_save+perm+'auc_'+name+'.pkl'
    with open(save_name, 'wb') as file:
        pickle.dump(auc, file)
        
##############################################################################
def leaveKout_CV(X, y, n_scz_te, rep, perms, classifiers, parameters, count,
                    freq_bands, x_size, auc, nz_coef_idx, nz_coef_val, n_BAitaSig = None, specificBands = []):
    
    skf = StratifiedKFold(n_splits=int(sum(y==0)//n_scz_te),shuffle=True) #, random_state = rep+2000)
    for tr_idx, te_idx in skf.split(X,y):
        # Compute test and train targets
        y_tr = np.ravel(y[tr_idx])
        y_te = np.ravel(y[te_idx])
        
        # Make gridsearch function
        clf_name = list(classifiers.keys())[0]
        count += 1
        for i in range(len(freq_bands)):
            if (freq_bands[i] not in specificBands) and (len(specificBands) > 0):
                continue
            
            clf = GridSearchCV(classifiers[clf_name], {'alpha' : parameters[freq_bands[i]]}, 
                       cv = StratifiedKFold(n_splits = int(sum(y_tr==0)//n_scz_te)), 
                       scoring = 'roc_auc', n_jobs = -1) #return_train_score=True,
            
            # Compute test and train sets 
            if n_BAitaSig == None:
                X_tr = X[tr_idx, x_size*i:x_size*(i+1)]
                X_te = X[te_idx, x_size*i:x_size*(i+1)]
            else:
                if x_size == sum(n_BAitaSig):
                    X_tr = X[tr_idx, :]
                    X_te = X[te_idx, :]
                else:
                    n_temp = [0]
                    n_temp.extend(np.cumsum(n_BAitaSig))
                    X_tr = X[tr_idx, n_temp[i]:n_temp[i+1]]
                    X_te = X[te_idx, n_temp[i]:n_temp[i+1]]

            #pdb.set_trace()
            # Standardize
            scaler_out = preprocessing.StandardScaler().fit(X_tr)
            X_tr =  scaler_out.transform(X_tr)
            X_te =  scaler_out.transform(X_te)

            # Fit data and save auc scores
            fit = clf.fit(X_tr, y_tr)
            auc[freq_bands[i]][count] = clf.score(X_te, y_te)
            
            # Save coefficients
            #coef_idx = np.nonzero(fit.best_estimator_.coef_)
            if len(perms) == 1:
                coef_idx = np.nonzero(fit.best_estimator_.coef_)
                nz_coef_idx[freq_bands[i]].append(coef_idx)
                nz_coef_val[freq_bands[i]].append(fit.best_estimator_.coef_[coef_idx])
    return auc, nz_coef_idx, nz_coef_val, count
           
##############################################################################
def CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters, n_BAitaSig=None, specificBands = []):
    """
    Parameters
    ----------
    X : np.array 
        Matrix with dimension (subjects)x(feature vector).
    y : np.array
        Vector with classifications (0: healthy, 1: schizo).
    n_scz_te : int
        Desired number of schizophrenic patients in each test set.
    reps : range(*)
        Range with desired number (*) of extra times the code should run.
    separate_bands : boolean
        True = seperate data into frequency bands. False = don't separate.
    perms : range(*)
        Range with desired number (*) of permutations. 
        *=1 indicates no permutations.
    dir_save : string
        Directory path to where the results should be saved.
    classifiers : dictionary
        Dictionary containing classifiers. E.g. {'lasso' : Lasso(max_iter = 10000)}
    parameters : dictionary
        Dictionary containing parameters to the classifiers as in "classifiers"

    Notes
    -------
    Saves three different values in the dir_save path: 
    auc : dictionary
        Contains the auc-scores for each loop, either divided into bands or 
        with the key "all".
    nz_coef_idx : dictionary
        Contains the non-zero coefficient indices for each loop, either 
        divided into bands or with the key "all".
    nz_coef_val : dictionary
        Contains the non-zero coefficient values (the weights) for each 
        loop, either divided into bands or with the key "all".
        
    """  
    
    # Get current date as day and month
    date = datetime.now().strftime("%d%m")
    
    # Check if data should be seperated into bands or not:
    if separate_bands:
        freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    else:
        freq_bands = ['all']
    
    # Choose what loop to use tqdm in
    if len(perms) > 1:
        y_org = y
        tqdm_perms = tqdm(perms)
        tqdm_reps = reps
    else: 
        tqdm_perms = perms
        tqdm_reps = tqdm(reps)
    
    # Initialize space for values 
    auc = {}
    nz_coef_idx= {}
    nz_coef_val= {}
    nb_loops = len(reps)*(sum(y==0)//n_scz_te)*len(perms)
    # Define the size of X
    x_size = int(X.shape[1]/len(freq_bands))
    for i in freq_bands:
        auc[i] = np.zeros(nb_loops)   # e.g. auc = {'delta':[] , 'theta': [], 'alpha': [], ....}
        nz_coef_idx[i] = []
        nz_coef_val[i] = []
    
    count = -1
    for perm in tqdm_perms:
        if len(perms) > 1:
            y = shuffle(y_org, random_state=perm).reset_index(drop=True)   
            
        for rep in tqdm_reps:
            # Run leave K out with grid search
           
            auc, nz_coef_idx, nz_coef_val, count = leaveKout_CV(X, y, n_scz_te, rep, 
                                            perms, classifiers, parameters, count, 
                                            freq_bands, x_size, auc, nz_coef_idx, 
                                            nz_coef_val, n_BAitaSig, specificBands)
                                
            # Save to computer 
            clf_name = list(classifiers.keys())[0]
            save_values(auc, nz_coef_idx, nz_coef_val, clf_name, perms, dir_save, date)
            
##############################################################################
def getEgillX(X):
    ## Extract rois (Egill)
    # Old rois with paracentral, used in the report
    #rois = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    # New rois with parahippocampal
    rois =  np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    
    rois = np.array([rois,rois]).reshape(-1,order='F')
    
    roi_mat = np.outer(rois,rois)
    
    # Calculate correlation    
    roi_vec = []
        
    for i in range(6):
        roi_vec.extend(list(roi_mat[np.triu_indices(len(roi_mat), k=1)]))
    
    Xnew = X[:,np.nonzero(roi_vec)]
    Xnew = np.reshape(Xnew,(len(Xnew),sum(roi_vec)))
    return Xnew

##############################################################################
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
def getEgillParameters(con_type, separate_bands):
    if separate_bands:
        if con_type == 'plv':
            #parameters= {'lasso' : {'alpha': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]}}
            parameters= {'delta': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                         'theta': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
                         'alpha': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                         'beta1': [0.06, 0.07, 0.08, 0.09, 0.1, 0.11], #[0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                         'beta2': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
                         'gamma': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]}
        
        elif con_type == 'pli':
            parameters = {'delta': [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14], 
                         'theta': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'alpha': [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
                         'beta1': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
                         'beta2': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
                         'gamma': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]}
        #[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        #[0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'lps':
            parameters = {'delta': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'theta': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
                         'alpha': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'beta1': [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'beta2': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
                         'gamma': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]}
        #[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'corr':
            parameters = {'delta': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13],
                         'theta': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13],
                         'alpha': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13],
                         'beta1': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13],
                         'beta2': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13],
                         'gamma': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13]}
        
        elif con_type == 'coherence':
            parameters = {'delta': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                         'theta': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
                         'alpha': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
                         'beta1': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
                         'beta2': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
                         'gamma': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]}
    else: 
        if con_type == 'plv':
            parameters= {'all': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]}
        
        elif con_type == 'pli':
            parameters= {'all': [0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18]}
        #[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        #[0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        #[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'lps':
            parameters= {'all':  [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]}
        #[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'corr':
            parameters= {'all': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]}
        
        elif con_type == 'coherence':
            parameters= {'all': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]}

    return parameters
       

##############################################################################  
def getBAitaSigParameters(con_type, separate_bands):
    if separate_bands:
        if con_type == 'plv':
            parameters= {'delta': [0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], #[0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], 
                         'theta': [0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                         'alpha': [0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                         'beta1': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                         'beta2': [0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                         'gamma': [0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}
        #[0.0001, 0.00033, 0.00066, 0.001, 0.0033, 0.0066, 0.01, 0.033, 0.066]
        #[0.001, 0.0025, 0.0055, 0.0075, 0.01, 0.025, 0.05, 0.075]
        #[0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        #[0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4]
        # [0.002, 0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4]
        elif con_type == 'pli':
            parameters= {'delta': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15],
                         'theta': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'alpha': [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'beta1': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                         'beta2': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                         'gamma': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]}
        #[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        #[0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'lps':
            parameters= {'delta': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
                         'theta': [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                         'alpha': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'beta1': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
                         'beta2': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14],
                         'gamma': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]}
        #[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
    else: 
        if con_type == 'plv':
            parameters= {'all': [0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}
        
        elif con_type == 'pli':
            parameters= {'all': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]}
        #[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        #[0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        #[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'lps':
            parameters= {'all':  [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]}
        #[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        # [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4,0.5]
            
    return parameters
 
##############################################################################  
def getBAitaParameters(con_type, separate_bands):
    if separate_bands:
        if con_type == 'plv':
            parameters= {'delta': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
                         'theta': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
                         'alpha': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
                         'beta1': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
                         'beta2': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
                         'gamma': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]}
        
        elif con_type == 'pli':
            parameters= {'delta': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'theta': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'alpha': [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
                         'beta1': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21],
                         'beta2': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
                         'gamma': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]}
        #[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        #[0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'lps':
            parameters= {'delta': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'theta': [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'alpha': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'beta1': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'beta2': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'gamma': [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]}
        #[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
    else: 
        if con_type == 'plv':
            parameters= {'all': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]}
        
        elif con_type == 'pli':
            parameters= {'all': [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21]}
        #[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        #[0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        #[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        elif con_type == 'lps':
            parameters= {'all':  [0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18]}
        #[0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            
    return parameters
 
##############################################################################   
##############################################################################   












