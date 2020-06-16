#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:39:26 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm #count ffor loops
import math
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn import preprocessing
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.linear_model import Lasso
from makeClassificationTest2 import getEgillX, getEgillParameters
from makeClassificationTest2 import getBAitaSigX, getBAitaSigParameters, getBAitaParameters
import seaborn as sns
from makeClassificationTest2 import getData
from utilsResults import getNewestFolderDate

import pdb
#{}
#[]

                
##############################################################################
def leaveKout_CV(X, y, n_scz_te, rep, perms, classifiers, parameters, count,
                    freq_bands, x_size, auc, nz_coef_idx, nz_coef_val, n_BAitaSig = None):
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
    
    skf = StratifiedKFold(n_splits=int(sum(y==0)//n_scz_te),shuffle=True, random_state = rep)
    count_plt = 0
    fig, ax = plt.subplots(2,3 , figsize=(10,6.5))
    for tr_idx, te_idx in skf.split(X,y):
        # Compute test and train targets
        y_tr = np.ravel(y[tr_idx])
        y_te = np.ravel(y[te_idx])
        
        # Make gridsearch function
        clf_name = list(classifiers.keys())[0]
        count += 1
        sns.set(font_scale=1.5)
        for i in range(1): #range(len(freq_bands)):
            if count_plt == 6:
                plt.suptitle('Example of line search for the regularization parameter', fontsize= 18)
                plt.tight_layout()
                plt.subplots_adjust(top = 0.84, bottom = 0.15, hspace = 0.5, wspace = 0.45)
                fig.legend(['Train', 'Validation'], bbox_to_anchor = (0.5, 0.89), 
                           borderaxespad = 0., loc = 'upper center', ncol = 2)
                
                plt.show()
                fig.savefig('/share/FannyMaster/PythonNew/Figures/LineSearchEx.jpg', bbox_inches = 'tight')
                sns.reset_orig()
                raise 
                
            i = 1
            clf = GridSearchCV(classifiers[clf_name], {'alpha' :parameters[freq_bands[i]]}, 
                       cv = StratifiedKFold(n_splits = int(sum(y_tr==0)//n_scz_te)), 
                       scoring = 'roc_auc', n_jobs = -1, return_train_score=True)
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
                
                
            # Standardize
            scaler_out = preprocessing.StandardScaler().fit(X_tr)
            X_tr =  scaler_out.transform(X_tr)
            X_te =  scaler_out.transform(X_te)

            # Fit data and save auc scores
            fit = clf.fit(X_tr, y_tr)
            auc[freq_bands[i]][count] = fit.score(X_te, y_te)
            
            # Make parameter plot
            #plot_grid_search(clf.cv_results_, 'score', parameters[freq_bands[i]], 'log($\lambda$) ' + freq_bands[i])
            cv_results = clf.cv_results_
            metric = 'score'
            grid_param_1 = parameters[freq_bands[i]]
            
            scores_mean = cv_results[('mean_test_' + metric)]
            # scores_sd = cv_results[('std_test_' + metric)]
            scores_mean_tr = cv_results[('mean_train_' + metric)]
            
            # Set plot style
            #plt.style.use('seaborn')
        
            # Plot Grid search scores

            sns.set(font_scale=1.5)
            df1 = pd.DataFrame({'log($\lambda$)':[math.log(i) for i in grid_param_1], 'CV Average AUC' : scores_mean_tr, 'type' : ['train']*len(scores_mean_tr)})
            df2 = pd.DataFrame({'log($\lambda$)':[math.log(i) for i in grid_param_1], 'CV Average AUC' : scores_mean, 'type' : ['test']*len(scores_mean_tr)})
            sns.lineplot(x = 'log($\lambda$)', y = 'CV Average AUC', style='type', legend = False, markers = "o", data = df1, ax = ax[count_plt//3][count_plt%3])
            sns.lineplot(x = 'log($\lambda$)', y = 'CV Average AUC', style='type', legend = False, markers = "o", data = df2, ax = ax[count_plt//3][count_plt%3])

            ax[count_plt//3][count_plt%3].set_xlabel('log($\lambda$)', fontsize=14)
            ax[count_plt//3][count_plt%3].set_ylabel('CV Average AUC' , fontsize=14) 
            
            #pprint(clf.cv_results_)
            #pdb.set_trace() # Type "exit" to get out, type "c" to continue
            count_plt += 1
            if len(perms) == 1:
                coef_idx = np.nonzero(fit.best_estimator_.coef_)
                nz_coef_idx[freq_bands[i]].append(coef_idx)
                nz_coef_val[freq_bands[i]].append(fit.best_estimator_.coef_[coef_idx])

    return auc, nz_coef_idx, nz_coef_val, count

##############################################################################
def CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
                  classifiers, parameters, n_BAitaSig = None):
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
    
    # Check if data should be seperated into bands or not:
    if separate_bands:
        freq_bands = ['delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma']
    else:
        freq_bands = ['all']
    
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
            auc, nz_coef_idx, nz_coef_val, count = leaveKout_CV(X, y, n_scz_te, rep, 
                                            perms, classifiers, parameters, count, 
                                            freq_bands, x_size, auc, nz_coef_idx, 
                                            nz_coef_val, n_BAitaSig)



#%%
con_type = 'lps'
separate_bands = True # False = All bands together
partialData = True

atlas = 'BAita' # DKEgill, BAita, BAitaSig

sns.set(font_scale=1.5)
freq_band_type = 'DiLorenzo'
# Directories
dir_folders = r'/share/FannyMaster/PythonNew/' + atlas + '_timeseries_'
newest_date = getNewestFolderDate(dir_folders)
dir_features = dir_folders + newest_date + '/' + freq_band_type + '/Features' 
dir_y_ID = r'/share/FannyMaster/PythonNew/Age_Gender.csv'
n_scz_te = 2
reps = range(1)
classifiers = {'lasso' : Lasso(max_iter = 10000)}  
dir_save = dir_folders + newest_date + '/' + freq_band_type + '/classificationResults/' + con_type.capitalize() 
X,y = getData(dir_features, dir_y_ID, con_type, partialData)

if atlas == 'DKEgill':
    X = getEgillX(X)
    n_BAitaSig = None
    parameters = getEgillParameters(con_type, separate_bands)
elif atlas == 'BAitaSig':
    X, n_BAitaSig = getBAitaSigX(X)
    parameters = getBAitaSigParameters(con_type, separate_bands)
elif atlas == 'BAita':
    parameters = getBAitaParameters(con_type, separate_bands)
    n_BAitaSig = None

perms = range(1) # 1 = No permutations
CV_classifier(X, y, n_scz_te, reps, separate_bands, perms, dir_save, 
              classifiers, parameters)




