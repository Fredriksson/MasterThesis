#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:47:45 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

import numpy as np
import os.path as op
import pickle
import pandas as pd
from tqdm import tqdm #count ffor loops
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import preprocessing
from datetime import datetime
from sklearn.utils import shuffle
from os import mkdir
# from jointUtils import getXy
import pdb #For debugging add pdb.set_trace() in function use c for continue, u for up, exit for exiting debug mode etc.

#{}
#[]


#%%
##############################################################################
# def getData(dir_features, dir_y_ID, con_type, partialData = False):
#     """
#     Parameters
#     ----------
#     dir_features : string
#         Directory path to where the features are saved.
#     dir_y_ID : string
#         Directory path to where the y-vector can be extracted.
#     con_type : string
#         The desired connectivity measure.
#     partialData : boolean (default False)
#         Used to chose wether the six  noisy subjects should be included or not.
#         False = the full data set

#     Returns
#     -------
#     X : array of arrays
#         Matrix containing a vector with all the features for each subject.
#         Dimension (number of subjects)x(number of features).
#     y : array
#         A vector containing the class-information. 
#         Remember: 1 = healty controls, 0 = schizophrenic

#     """
   
#     # Make directory path and get file    
#     file_path = dir_features + '/feature_dict_' + con_type + '.pkl'
#     with open(file_path, 'rb') as file:
#         feature_dict = pickle.load(file)
    
#     # Load csv with y classes
#     try:
#         dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
#         if partialData:
#             dat = dat[~dat['Id'].isin(['D950', 'D935', 'D259', 'D255', 'D247', 'D160'])]    
#     except: 
#         dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Group'], sep = ';')

    
    
#     X = []
#     y = []
#     for i, row in dat.iterrows():
#         X.append(feature_dict['features'][row['Id']])
#         y.append(row['Group'])
#     X = np.array(X)
#     y = 1 - pd.Series(y)
#     #pdb.set_trace()
#     return X, y

# def getData_PECANS2(dir_features, dir_y_ID, con_type, partialData = False):
#     """
#     Parameters
#     ----------
#     dir_features : string
#         Directory path to where the features are saved.
#     dir_y_ID : string
#         Directory path to where the y-vector can be extracted.
#     con_type : string
#         The desired connectivity measure.
#     partialData : boolean (default False)
#         Used to chose wether the six  noisy subjects should be included or not.
#         False = the full data set

#     Returns
#     -------
#     X : array of arrays
#         Matrix containing a vector with all the features for each subject.
#         Dimension (number of subjects)x(number of features).
#     y : array
#         A vector containing the class-information. 
#         Remember: 1 = healty controls, 0 = schizophrenic

#     """
   
#     # Make directory path and get file    
#     file_path = dir_features + '/feature_dict_' + con_type + '.pkl'
#     with open(file_path, 'rb') as file:
#         feature_dict = pickle.load(file)
    
#     # Load csv with y classes
#     dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id','Group'])
#     if partialData:
#         dat = dat[~dat['Id'].isin(['D950', 'D935', 'D259', 'D255', 'D247', 'D160'])]
    
    
#     X = []
#     y = []
#     for i, row in dat.iterrows():
#         X.append(feature_dict['features'][row['Id']])
#         y.append(row['Group'])
#     X = np.array(X)
#     y = 1 - pd.Series(y)
#     #pdb.set_trace()
#     return X, y


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
    """
    Calculates the leave K out cross validation. 

    Parameters
    ----------
    X : array of arrays
        Matrix containing a vector with all the features for each subject.
        Dimension (number of subjects)x(number of features).
    y : array
        A vector containing the class-information. 
        Remember: 1 = healty controls, 0 = schizophrenic
        
    n_scz_te : int
        Desired number of schizophrenic patients in each test set.
    rep : integer
        The number of repition that has been used so far.
    perms : range(*)
        Range with desired number (*) of permutations. 
        *=1 indicates no permutations.
    classifiers : dictionary
        Dictionary containing classifiers. E.g. {'lasso' : Lasso(max_iter = 10000)}
    parameters : dictionary
        Dictionary containing parameters to the classifiers as in "classifiers"
    count : integer
        Used to know how many loops that have been made due to the pre 
        allocated space for AUC.
    freq_bands : list of strings
        Either ['all'] or ['detla','theta','alpha','beta1','beta2','gamma'].
    x_size : integer
        The size each X has which changes depending on freq_bands.
    auc : dictionary
        Contains the auc-scores for each loop, either divided into bands or 
        with the key "all".
    nz_coef_idx : dictionary
        Contains the non-zero coefficient indices for each loop, either 
        divided into bands or with the key "all".
    nz_coef_val : dictionary
        Contains the non-zero coefficient values (the weights) for each 
        loop, either divided into bands or with the key "all".
    n_BAitaSig : list of integers, optional
        The number of connections in each band when BAitaSig is used. 
        The default is None.
    specificBands : list of string, optional
        Can be used if only specific bands are desired. 
        The default is [].

    Returns
    -------
    auc : dictionary
        Contains the updated auc-scores for each loop, either divided into 
        bands or with the key "all".
    nz_coef_idx : dictionary
        Contains the updated non-zero coefficient indices for each loop, 
        either divided into bands or with the key "all".
    nz_coef_val : dictionary
        Contains the updated non-zero coefficient values (the weights) for 
        each loop, either divided into bands or with the key "all".
    count : integer
        Used to know how many loops that have been made due to the pre 
        allocated space for AUC.

    """
    
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
    n_BAitaSig : list of integers, optional
        The number of connections in each band when BAitaSig is used. 
        The default is None.
    specificBands : list of string, optional
        Can be used if only specific bands are desired. 
        The default is [].

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
    """
    Extracts the areas that Egill picked out from the Desikan-Killiany atlas.

    Parameters
    ----------
    X : list of lists
        Full data set for the Desikan-Killiany time series.

    Returns
    -------
    Xnew : list of lists
        Only contains the time series of the areas that Egill picked out.

    """

    # Old rois with paracentral (used in the report)
    # rois = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    # New rois with parahippocampal
    rois =  np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    rois = np.array([rois,rois]).reshape(-1,order='F')
    
    # Make matrix with zeros and ones
    roi_mat = np.outer(rois,rois)
    
    # Make an vector out of the upper triangle of the roi_mat 
    roi_vec = []       
    for i in range(6):
        roi_vec.extend(list(roi_mat[np.triu_indices(len(roi_mat), k=1)]))
    
    Xnew = X[:,np.nonzero(roi_vec)]
    Xnew = np.reshape(Xnew,(len(Xnew),sum(roi_vec)))
    return Xnew

##############################################################################
def BAitaSig():
    """
    Gets Di Lorenzo et al's significant areas and saves them as a vector.

    Returns
    -------
    roi_vec : list
        Contains a boolean vector with Di Lorenzo et al.'s significant pairs.
    n_BAitaSig : list of integers
        The number of connections in each band when BAitaSig is used. 

    """
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
        roi_mat = abs(dat)>3.485124
        roi_flat = list(np.array(roi_mat)[np.triu_indices(len(roi_mat), k=1)])
        roi_vec.extend(roi_flat)

        n_BAitaSig.extend([sum(roi_flat)])
    return roi_vec, n_BAitaSig
    
##############################################################################
def significant_connected_areasBAitaSigX(X):
    """
    The significant connections found by Di Lorenzo et al. 

    Parameters
    ----------
    X : list of lists
        Full data set wiht time series for the Di Lorenzo inspired atlas.

    Returns
    -------
    Xnew : list of lists
        Only contains the time series of the areas that Di Lorenzo et al 
        rendered significant.
    n_BAitaSig : list of integers
        The number of connections in each band when BAitaSig is used. 

    """
    ## Extract rois (Brodmann Areas collected as in Di Lorenzo et al.)
    
    roi_vec, n_BAitaSig = BAitaSig()
    
    Xnew = X[:,np.nonzero(roi_vec)]
    Xnew = np.reshape(Xnew,(len(Xnew),sum(roi_vec)))
    return Xnew, n_BAitaSig 

##############################################################################
def getEgillParameters(con_type, separate_bands):
    """
    The optimized intervals used for the DKEgill data set

    Parameters
    ----------
    con_type : string
        The desired connectivity measure.
    separate_bands : boolean
        True = seperate data into frequency bands. False = don't separate.

    Returns
    -------
    parameters : dictionary
        The parameters that should be used for each frequency band.

    """
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
    """
    The optimized intervals used for the BAitaSig data set

    Parameters
    ----------
    con_type : string
        The desired connectivity measure.
    separate_bands : boolean
        True = seperate data into frequency bands. False = don't separate.

    Returns
    -------
    parameters : dictionary
        The parameters that should be used for each frequency band.

    """
    
    if separate_bands:
        if con_type == 'plv':
            parameters= {'delta': [0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], #[0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], 
                         'theta': [0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                         'alpha': [0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                         'beta1': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                         'beta2': [0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
                         'gamma': [0.006, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}

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

        elif con_type == 'lps':
            parameters= {'all':  [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]}
            
    return parameters
 
##############################################################################  
def getBAitaParameters(con_type, separate_bands):
    """
    The optimized intervals used for the BAita data set

    Parameters
    ----------
    con_type : string
        The desired connectivity measure.
    separate_bands : boolean
        True = seperate data into frequency bands. False = don't separate.

    Returns
    -------
    parameters : dictionary
        The parameters that should be used for each frequency band.

    """
    
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
            
        elif con_type == 'lps':
            parameters= {'delta': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'theta': [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
                         'alpha': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'beta1': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'beta2': [0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17],
                         'gamma': [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]}
        
    else: 
        if con_type == 'plv':
            parameters= {'all': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]}
        
        elif con_type == 'pli':
            parameters= {'all': [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21]}
  
        elif con_type == 'lps':
            parameters= {'all':  [0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18]}
            
    return parameters
 
##############################################################################   
##############################################################################   






