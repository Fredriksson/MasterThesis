#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:02:44 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""


#%%
from mne import read_labels_from_annot
from mne.datasets import fetch_fsaverage
import numpy as np
import pandas as pd
import os.path as op
import pickle
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import OrderedDict
import nibabel as nb
from nilearn import datasets, plotting
from nilearn import surface
import pdb
import matplotlib as mpl

# {}
# []

#%% Functions 
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
def band_idx(idx, n_feature_bands, n_BAitaSig=None):
    """
    Parameters
    ----------
    idx : int
        Index number in vector containing several bands.
    n_feature_bands : int
        Number of feature bands.

    Returns
    -------
    A list with two elements:
        band : int
            Band number.
        idx_real : int
            The real index number.
    """
    if n_BAitaSig == None:
        for band in range(6):
            if (idx in range(band*n_feature_bands, (band+1)*n_feature_bands)):
                idx_real = idx % n_feature_bands
                return [band, idx_real]
    else: 
        n_cumsum = [0]
        n_cumsum.extend(np.cumsum(n_BAitaSig))
        for band, cum_val in enumerate(n_cumsum[1:]):
            if (idx < cum_val):
                idx_real = idx - n_cumsum[band]
                return [band, idx_real]

        
        
        
##############################################################################  
from makeClassification import BAitaSig      
def getLabels(atlas): #getDKLabels
    """
    Returns
    -------
    A list containing Desikan-Killiany brain areas. Hence each element of the 
    list is a string with the name of one area. 
    """
    
    if atlas == 'DKEgill':
        # Desikan-Killiany Atlas 
        fs_dir = fetch_fsaverage(verbose=True)
        subjects_dir = op.dirname(fs_dir) 
        #fs_dir = '/home/kmsa/mne_data/MNE-fsaverage-data/fsaverage' 
        #subjects_dir = '/home/kmsa/mne_data/MNE-fsaverage-data'
        subject = 'fsaverage'
        
        #parc = 'PALS_B12_Brodmann' 
        parc = 'aparc'
        labels = read_labels_from_annot(subject, parc = parc, hemi='both',
                                         surf_name= 'white', annot_fname = None, regexp = None,
                                         subjects_dir = subjects_dir, verbose = None)
        
        
        labels = labels[:-1] #[] 
        labels = [i.name for i in labels]
        
        #rois = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]
        #rois = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        rois = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
        rois = np.array([rois, rois]).reshape(-1,order='F')
        
        labels = np.array(labels)[np.nonzero(rois)] 
        return labels
        
    elif atlas == 'DK':
        # Desikan-Killiany Atlas 
        fs_dir = fetch_fsaverage(verbose=True)
        subjects_dir = op.dirname(fs_dir) 
        #fs_dir = '/home/kmsa/mne_data/MNE-fsaverage-data/fsaverage' 
        #subjects_dir = '/home/kmsa/mne_data/MNE-fsaverage-data'
        subject = 'fsaverage'
        
        #parc = 'PALS_B12_Brodmann' 
        parc = 'aparc'
        labels = read_labels_from_annot(subject, parc = parc, hemi='both',
                                         surf_name= 'white', annot_fname = None, regexp = None,
                                         subjects_dir = subjects_dir, verbose = None)
        labels = labels[:-1] 
        return labels
    
    elif atlas == 'BAita':
        ita_labels = ['SMA', 'SPL', 'SFC', 'AFC', 'OFC', 'LFC', #'INS',
                     'LTL', 'ACC', 'PCC', 'PHG', 'IPL', 'FLC', 'PVC']
        labels = np.array([[i+'-lh',i+'-rh'] for i in ita_labels]).reshape(-1)
        return labels
    
    elif atlas == 'BAitaSig':
        ## Extract rois (Brodmann Areas collected as in Di Lorenzo et al.)
        
        roi_vec, n_BAitaSig = BAitaSig() 
        
        ita_labels = ['SMA', 'SPL', 'SFC', 'AFC', 'OFC', 'LFC', #'INS',
                         'LTL', 'ACC', 'PCC', 'PHG', 'IPL', 'FLC', 'PVC']
        ita_labels = np.array([[i+'-lh',i+'-rh'] for i in ita_labels]).reshape(-1)
        
        # Get indices of triangular matrix
        x, y = np.triu_indices(len(ita_labels),k=1)
        labels = []
        for i in range(6):
            rois = roi_vec[len(roi_vec)//6*i : len(roi_vec)//6*(i+1)]
            sig_idx = np.nonzero(rois)
            
            labels.append([ita_labels[x[j]]+'_' +ita_labels[y[j]] for j in sig_idx[0]])

        return labels, n_BAitaSig
    
    

##############################################################################
def getData(dirData, wanted_info, clf_types):
    """
    Parameters
    ----------
    dirData : string
        Directory path to where the data is saved.
    wanted_info : string
        Can be 'all', 'bands' or 'both'.
    clf_types : list of strings
        Each string corresponds to the name of a classifier.

    Returns
    -------
    data_dict : dictionary
        Dictionary with the data that is placed in dirData.

    """
    
    for clf in clf_types:
        if wanted_info == 'both':
            # Get 'bands' data
            file_paths = glob(dirData + clf +'_bands*.pkl')
            newest_file = max(file_paths, key=op.getctime)
            with open(newest_file, 'rb') as file:
                data_dict_bands = pickle.load(file)
            
            # Get 'all' data
            file_paths = glob(dirData + clf + '_all*.pkl')
            newest_file = max(file_paths, key=op.getctime)
            with open(newest_file, 'rb') as file:
                data_dict = pickle.load(file)
            
            # Merge the two dictionaries
            data_dict.update(data_dict_bands)
            
        elif wanted_info == 'all':
            # Get 'all' data
            file_paths = glob(dirData + clf + '_all*.pkl')
            newest_file = max(file_paths, key=op.getctime)
            with open(newest_file, 'rb') as file:
                data_dict = pickle.load(file)
        
        elif wanted_info == 'bands':
            # Get 'bands' data
            file_paths = glob(dirData + clf + '_bands*.pkl')
            newest_file = max(file_paths, key=op.getctime)
            with open(newest_file, 'rb') as file:
                data_dict = pickle.load(file)
                
        else:
            raise NameError('The wanted_info "' + wanted_info + '" is not valid')
            
    return data_dict
    
##############################################################################
def most_connected_areas(dir_nz_coef_idx, min_fraction, labels, wanted_info, clf_types, n_BAitaSig=None):
    """
    Parameters
    ----------
    dir_nz_coef_idx : string
        Path to the wanted non-zero coefficients indices. 
    min_fraction : float
        The minimum fraction of times a feature have been used.
    labels : list
        A containig strings with names of brain areas, generated by 
        get*Labels().
    wanted_info : string
        Can be 'all', 'bands' or 'both'. 
    clf_types : list of strings
        Each string corresponds to the name of a classifier.
        
    Returns
    -------
    connected_areas : string
        String containing the most connected areas and what band that is used.

    Notes
    -------
    Prints the areas that are connected more then the minimum fraction of 
    times.

    """
    
    # Get data  
    coef_idx_dict = getData(dir_nz_coef_idx, wanted_info, clf_types)
    # Used frequency bands 
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    # Initialize dictionary to return
    connected_areas = OrderedDict()
    
    if 'Partial' in dir_nz_coef_idx:
        denom = 260
    else: 
        denom = 280
    
   
    # Get indices of triangular matrix
    x, y = np.triu_indices(len(labels),k=1)  
    
    # Number of elements for each frequency band 
    n_feature_bands = int(((len(labels))*(len(labels)-1))/2)

    for j, band in enumerate(list(coef_idx_dict.keys())) :
        coef_idx = coef_idx_dict[band]
        # Get top recurrent index values
        coef_idx_vec = pd.DataFrame(pd.core.common.flatten(coef_idx))
        min_count = len(coef_idx)*min_fraction # Multiply with fraction to gain percentage
        coef_idx_count = coef_idx_vec[0].value_counts()
        coef_idx_top = coef_idx_count[coef_idx_count >= min_count].index.tolist()
        
        # Get band number and real index
        band_index = [band_idx(idx, n_feature_bands, n_BAitaSig) for idx in coef_idx_top]
        #pdb.set_trace()
        
        if n_BAitaSig == None:
            #Print names of most connected brain areas
            if band=='all': # If all bands are used 
                con_area = [[freq_bands[i[0]], labels[x[i[1]]], labels[y[i[1]]], np.round(coef_idx_count.values[idx]/denom, 4)*10] for idx, i in enumerate(band_index)]
                connected_areas[band] = con_area
            else: # If the bands are seperated
                con_area = [[labels[x[i[1]]],labels[y[i[1]]], np.round(coef_idx_count.values[idx]/denom*10, 4)*10] for idx, i in enumerate(band_index)]
                connected_areas[band] = con_area
        else:
            #Print names of most connected brain areas
            if band=='all': # If all bands are used 
                con_area = [[freq_bands[i[0]], labels[i[0]][i[1]], np.round(coef_idx_count.values[idx]/denom*10, 4)*10] for idx, i in enumerate(band_index)]
                connected_areas[band] = con_area
            else: # If the bands are seperated
                con_area = [[labels[i[0]][i[1]], np.round(coef_idx_count.values[idx]/denom*10, 4)*10] for idx, i in enumerate(band_index)]
                connected_areas[band] = con_area        
            
    return connected_areas

##############################################################################


def boxplots_auc(dir_auc, dir_save, clf_types, wanted_info, figsize, con_type, atlas):
    """
    Makes boxplots of the auc-scores for the differnet bands, and/or for 
    different classifier methods. 
    
    Parameters
    ----------
    dir_auc : string
        Directory path to the auc files.
    dir_save : string
        Directory path to where the boxplot-figure is saved.
    clf_types : list of strings
        Each string is the name of a wanted classifer. E.g. ['lasso', 'svm']
    wanted_info : string
        Can be 'all', 'bands' or 'both'. 
    figsize : tuple
        The wanted figure size. E.g. (10,6)
        
    Returns
    -------
    fig : figure
        The constructed figure.

    """
    auc_plot = []
    xlabel = []
    clf_name = ''
    for clf in clf_types:
        auc_dict = getData(dir_auc, wanted_info, clf_types)
        clf_name += clf
    #auc_dict = getData(dir_auc, wanted_info)        

        #pdb.set_trace()
    
    # Used frequency bands 
    #freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
        freq_bands = list(auc_dict.keys())
        
        for band in freq_bands :
            auc = auc_dict[band]
            auc_plot.append(auc)
            
        df=pd.DataFrame.from_dict(auc_dict)
        #pdb.set_trace()
        if len(clf_types)>1:
            xlabel = [clf + ' ' + str(i) for i in freq_bands]
        else: 
            xlabel = freq_bands
            
    # pdb.set_trace()
    sns.set(font_scale=2)
    fig = plt.figure(figsize=figsize)
    if atlas == 'BAitaSig':
        atlas = 'DL-sig'
    elif atlas == 'BAita':
        atlas = 'DL'
    else: 
        atlas = 'DK-expert'
        
    sns.boxplot(data=df).set_title(atlas + ': Boxplots of ' + str(len(auc)) + ' AUC scores - ' + con_type.upper() )
    # plt.boxplot(auc_plot)
    # locs, junk = plt.xticks()
    # plt.xticks(locs, xlabel)
    plt.xlabel('Frequency Bands')
    plt.ylabel('AUC')
    # plt.title('Boxplots of ' + str(len(auc)) + ' AUC scores - ' + con_type )
    plt.show()
    
    date = datetime.now().strftime("%d%m")
    fig.savefig(dir_save + '/Boxplots_auc_' + con_type + '_' + wanted_info + '.jpg', bbox_inches = 'tight')
    sns.reset_orig()
    return fig    

##############################################################################
def getDKlabelCoordinates():
    """
    Gets the Desikan-Killiany coordinates, from the freesurfer package, 
    connected to each DK brain area. Returns a dictionary containing the brain 
    areas saved as the DK names presented in fsaverage, with corresponding
    coordinates.

    Returns
    -------
    label_coord : dictionary
        With the keys being the DK brain areas and the values being the 
        corresponding coordinates.

    """
    data_dir = '/share/FannyMaster/nilearn_data'
    atlas = 'desikan_killiany'
    #atlas = 'PALS_B12_Brodmann'
    annots = ['lh.aparc.annot', 'rh.aparc.annot']

    annot_left = nb.freesurfer.read_annot(op.join(data_dir,atlas,annots[0]))
    annot_right = nb.freesurfer.read_annot(op.join(data_dir,atlas,annots[1]))
    
    labels = annot_left[2]
    dk_atlas = {'map_left' : annot_left[0], 'map_right' : annot_right[0]}
    
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage',data_dir=data_dir)
    
    label_coord = {}
    for hemi in ['left', 'right']:
        vert = dk_atlas['map_%s' % hemi]
        coords, _ = surface.load_surf_mesh(fsaverage['pial_%s' % hemi])
        for idx, label in enumerate(labels):
            if ("unknown" not in str(label) and "corpuscallosum" not in str(label)):  # Omit the unknown and corpuscallosum labels.
                # Compute mean location of vertices in label of index idx
                if hemi=='left':
                    label_coord[str(label).split(sep="'")[1]+'-lh'] = np.mean(coords[vert == idx], axis=0)
                else:
                    label_coord[str(label).split(sep="'")[1]+'-rh'] = np.mean(coords[vert == idx], axis=0)
                
    return label_coord

##############################################################################
def plotConnections(cons_W_tot, title, atlas):
    """
    Plots the connections between different brain parts for the different 
    frequency bands on an image of the brain. The thickness of the connections 
    depends on how the weights have been calculated. Choose a title to explain 
    what weights that are used. 

    Parameters
    ----------
    cons_W_tot : list of lists
        Contains lists corresponding to each frequency band with the weights 
        between each used brain area.
    title : string
        The title you want for the figure.

    """
    
    # Get the coordinates for each brain area
    coordinates = getCoordinates(atlas)
    #pdb.set_trace()
    #Make symmetric
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    fig, ax = plt.subplots(2,3)
    for i in range(len(cons_W_tot)):
        cons_W = cons_W_tot[i]
        # cons_Wcount = cons_Wcount_tot[i]
        
        
        #Plot matrix
        #plotting.plot_matrix(cons_Wmu)
        #plotting.show()
        
        #Plot matrix
        #plotting.plot_matrix(cons_Wcount)
        #plotting.show()
        
        #Plot connections/important features (weights)
        plotting.plot_connectome(cons_W, coordinates, display_mode='z',
                                 node_color='k', node_size=1,
                                 title= freq_bands[i], axes = ax[i//3][i%3])
        #plotting.plot_connectome(cons_Wcount, coordinates, display_mode='z',
        #                         node_color='k', node_size=1, axes = ax[i%3][i//3+2])#title= freq_bands[i],

                                     
    fig.suptitle(title)
        #ax[i//3][i%3].plottt
    #plotting.show()
        
        
# con_fig = ax[i//3][i%3].imshow(con_mat, cmap='viridis', vmin= -1, vmax= 1)#, clim= np.percentile(con_mat[5, 95]))
    #     #ax[0][0].tight_layout()
    #     #plt.colorbar(con_fig)
    #     ax[i//3][i%3].set_title(freq_bands[i])
    
    
    # plt.tight_layout(pad = 0.6)
    # fig.suptitle(con_type.capitalize())
    # fig.subplots_adjust(top=0.88)
    # fig.colorbar(con_fig, ax = ax.ravel().tolist(), shrink = 0.98, ticks = [-1, 0, 1])
    # plt.show()
    
##############################################################################
def getWeights(coef_idx_top, coef_idx_vec, coef_val_vec):
    """
    Calculates two different type of weights: mu and count. Mu is the mean 
    value of the feature values used across the different cross validation 
    loops. Count is the amount of times a feature have been used across the 
    different cross validation loops. 

    Parameters
    ----------
    coef_idx_top : list
        The top used non-zero coefficient indices from the cross-validation.
    coef_idx_vec : list
        The non-zero coefficient indices from the cross-validation.
    coef_val_vec : list
        The top used non-zero coefficient values from the cross-validation..

    Returns
    -------
    mu_w : list
        The mean value of the top used feature values.
    cv_count : list
        The amount of times a feature have been used.

    """
    mu_w = []
    cv_count = []
    for k in coef_idx_top:
        sum_w = 0
        count = 0
        cv = 0
        for j in range(len(coef_idx_vec)):
            if k==coef_idx_vec[0][j]:
                sum_w+=coef_val_vec[0][j]
                count+=1
                cv+=1
        mu_w.append(sum_w/count)
        cv_count.append(cv)
    return mu_w, cv_count

##############################################################################
def makeConMatrix(band_index, mu_w, cv_count, j, atlas, wanted_info):
    """
    Makes the connectivity matrix for one given band. 

    Parameters
    ----------
    band_index : list of lists
        Each inner list contains a band number and an index number.
    mu_w : list
        The mean value of the top used feature values.
    cv_count : list
        The amount of times a feature have been used.
    j : int
        For 'all' this is a frquency band number, for 'bands' this should be 
        -1.
    x : list
        x-coordinates of a triangular matrix.
    y : list
        y-coordinates of a triangular matrix.

    Returns
    -------
    cons_Wmu : list
        Connectivity matrix with the mean value weights: Wmu.
    cons_Wcount : list
        Connectivity matrix with the counts as weights: Wmcount.

    """
    
    #Number of elements for each frequency band
    if atlas == 'DKEgill':
        label_size = 24
    else:
        label_size = 26
    
    #Get indices of triangular matrix
    x, y = np.triu_indices(label_size,k=1)
    
    roi_vec, n_BAitaSig = BAitaSig() 
    
    cons_Wmu = np.zeros([label_size,label_size])
    cons_Wcount = np.zeros([label_size,label_size])
    for count, i in enumerate(band_index):
        # When the bands are seperated from start. 'bands' is used. 
        if wanted_info == 'bands':
            if atlas == 'BAitaSig':
                rois = roi_vec[len(roi_vec)//6*i[0] : len(roi_vec)//6*(i[0]+1)]
                sig_idx = np.nonzero(rois)
                
                cons_Wmu[x[sig_idx[0][i[1]]], y[sig_idx[0][i[1]]]] = mu_w[count] #Assign mean feature weight
                cons_Wcount[x[sig_idx[0][i[1]]], y[sig_idx[0][i[1]]]] = cv_count[count] #Assign feature count
                
            else:
                cons_Wmu[x[i[1]], y[i[1]]] = mu_w[count] #Assign mean feature weight
                cons_Wcount[x[i[1]], y[i[1]]] = cv_count[count] #Assign feature count
            
        else:
        # When 'all' is used
            if i[0]==j:
                
                if atlas == 'BAitaSig':
                    rois = roi_vec[len(roi_vec)//6*i[0] : len(roi_vec)//6*(i[0]+1)]
                    sig_idx = np.nonzero(rois)
                
                    cons_Wmu[x[sig_idx[0][i[1]]], y[sig_idx[0][i[1]]]] = mu_w[count] #Assign mean feature weight
                    cons_Wcount[x[sig_idx[0][i[1]]], y[sig_idx[0][i[1]]]] = cv_count[count] #Assign feature count
                
                else:
                    cons_Wmu[x[i[1]], y[i[1]]] = mu_w[count] #Assign mean feature weight
                    cons_Wcount[x[i[1]], y[i[1]]] = cv_count[count] #Assign feature count
            
    cons_Wmu += cons_Wmu.T
    cons_Wcount += cons_Wcount.T
    return cons_Wmu, cons_Wcount


#     if n_BAitaSig == None:
#         for band in range(6):
#             if (idx in range(band*n_feature_bands, (band+1)*n_feature_bands)):
#                 idx_real = idx % n_feature_bands
#                 return [band, idx_real]
#     else: 
#         n_cumsum = [0]
#         n_cumsum.extend(np.cumsum(n_BAitaSig))
#         for band, cum_val in enumerate(n_cumsum[1:]):
#             if (idx < cum_val):
#                 idx_real = idx - n_cumsum[band]
#                 return [band, idx_real]

##############################################################################
def getConMatrices(dir_nz_coef_idx, dir_nz_coef_val, wanted_info, clf_types, min_fraction, atlas):
    """
    The main function to get the connectivity matrices for plotting. Gives two
    matrices containing different type of weights. One is the mean feature 
    weight, mWu, and the other is the number of occurrences weight, Wcount.

    Parameters
    ----------
    dir_nz_coef_idx : string
        Path to the wanted non-zero coefficients indices.
    dir_nz_coef_val : string
        Path to the wanted non-zero coefficients values.
    wanted_info : string
        Can be 'all', 'bands' or 'both'.
    clf_types : list of strings
        Each string is the name of a wanted classifer. E.g. ['lasso', 'svm']
    min_fraction : float
        The minimum fraction of times a feature have been used.

    Returns
    -------
    cons_Wmu_tot : list of lists
        Each inner list contains a connectivity matrix with muW as weights 
        for a given frequency band.
    cons_Wcount_tot : list of lists
        Each inner list contains a connectivity matrix with countW as weights 
        for a given frequency band.

    """
    # Used frequency bands 
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    # Get data    
    coef_idx_dict = getData(dir_nz_coef_idx, wanted_info, clf_types)
    coef_val_dict = getData(dir_nz_coef_val, wanted_info, clf_types)
    
    #pdb.set_trace()
    #Number of elements for each frequency band
    if atlas == 'BAitaSig':
        labels, n_BAitaSig = getLabels(atlas)
    else: 
        labels = getLabels(atlas)
        n_BAitaSig = None
    n_feature_bands = int(((len(labels))*(len(labels)-1))/2)
    
    
    # Used frequency bands 
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    
    cons_Wmu_tot = []
    cons_Wcount_tot = []
    for band in list(coef_idx_dict.keys()) :
        coef_idx = coef_idx_dict[band]
        coef_val = coef_val_dict[band]
        
        # Get top recurrent index values
        coef_idx_vec = pd.DataFrame(pd.core.common.flatten(coef_idx))
        min_count = len(coef_idx)*min_fraction # Multiply with fraction to gain percentage
        coef_idx_count = coef_idx_vec[0].value_counts()
        coef_idx_top = coef_idx_count[coef_idx_count >= min_count].index.tolist()
        
        coef_val_vec = pd.DataFrame(pd.core.common.flatten(coef_val))
                       
        # Get band number and real index
        band_index = [band_idx(idx, n_feature_bands, n_BAitaSig) for idx in coef_idx_top]
        #pdb.set_trace()
        # Get weights 
        mu_w, cv_count = getWeights(coef_idx_top, coef_idx_vec, coef_val_vec)
        
        
        #Get conmatrix all
        if band == 'all':
            for j in range(len(freq_bands)):
                cons_Wmu, cons_Wcount = makeConMatrix(band_index, mu_w, cv_count, j, atlas, wanted_info)
                cons_Wmu_tot.append(cons_Wmu)
                cons_Wcount_tot.append(cons_Wcount)
        else: 
            cons_Wmu, cons_Wcount = makeConMatrix(band_index, mu_w, cv_count, -1, atlas, wanted_info)
            cons_Wmu_tot.append(cons_Wmu)
            cons_Wcount_tot.append(cons_Wcount)
            
    return cons_Wmu_tot, cons_Wcount_tot

##############################################################################
def getBAlabelCoordinates():
    """
    Gets the Desikan-Killiany coordinates, from the freesurfer package, 
    connected to each DK brain area. Returns a dictionary containing the brain 
    areas saved as the DK names presented in fsaverage, with corresponding
    coordinates.

    Returns
    -------
    label_coord : dictionary
        With the keys being the DK brain areas and the values being the 
        corresponding coordinates.

    """
    
    data_dir = '/share/FannyMaster/mne_data/MNE-fsaverage-data/fsaverage/label/'
    annots = ['lh.PALS_B12_Brodmann.annot', 'rh.PALS_B12_Brodmann.annot']
    
    annot_left = nb.freesurfer.read_annot(op.join(data_dir,annots[0]))
    annot_right = nb.freesurfer.read_annot(op.join(data_dir,annots[1]))
    
    # dk_atlas = {'map_left' : annot_left[0], 'map_right' : annot_right[0],
    #             'labels_left' : annot_left[2], 'labels_right' : annot_right[2]}
    
    idx_ba_lh = {}
    for idx, lab in enumerate(annot_left[2]):
        if ('Brodmann.' in str(lab)):
            idx_ba_lh[str(lab).split(sep="'")[1]+'-lh'] = idx
    
    idx_ba_rh = {}
    for idx, lab in enumerate(annot_right[2]):
        if ('Brodmann.' in str(lab)):
            idx_ba_rh[str(lab).split(sep="'")[1]+'-rh'] = idx
    
    vert_lh = annot_left[0]
    vert_rh = annot_right[0]

    
    
    data_dir = '/share/FannyMaster/nilearn_data'
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage',data_dir=data_dir)

    coords_lh, _ = surface.load_surf_mesh(fsaverage['pial_left'])
    coords_rh, _ = surface.load_surf_mesh(fsaverage['pial_right'])

    ita_ba = [[1,2,3,4], [5,7], [6,8], [9,10], [11,47], [44,45,46], #[13],
           [20,21,22,38,41,42], [24,25,32, 33], #[24,25,32,33], 
           [23,29,30,31],
           [27,28,35,36], #[27,28,34,35,36], 
           [39,40,43], [19,37], [17,18]]
    
    ita_label = ['SMA', 'SPL', 'SFC', 'AFC', 'OFC', 'LFC', #'INS',
                 'LTL', 'ACC', 'PCC', 'PHG', 'IPL', 'FLC', 'PVC']
    label_coord = {}
    for idx, i in enumerate(ita_ba):
        lab_sum_lh = 0
        lab_sum_rh = 0
        count_lh = 0
        count_rh = 0
        for j in i:
            ba_lh = 'Brodmann.' + str(j) +'-lh'
            ba_rh = 'Brodmann.' + str(j) +'-rh'
            
            #if sum(vert_lh == idx_ba_lh[ba_lh])==0 or sum(vert_rh == idx_ba_rh[ba_rh])==0:
            #    pdb.set_trace()
            lh = vert_lh == idx_ba_lh[ba_lh]
            rh = vert_rh == idx_ba_rh[ba_rh]
            lab_sum_lh += np.sum(coords_lh[lh], 0)
            count_lh += sum(lh)
            lab_sum_rh += np.sum(coords_rh[rh], 0)
            count_rh += sum(rh)
            
        lab_sum_lh /= count_lh
        lab_sum_rh /= count_rh
        label_coord[ita_label[idx] + '-lh'] = lab_sum_lh
        label_coord[ita_label[idx] + '-rh'] = lab_sum_rh
   
   
    
    # label_coord = {}
    # for hemi in ['left', 'right']:
    #     labels = dk_atlas['labels_%s' % hemi]
    #     vert = dk_atlas['map_%s' % hemi]
    #     coords, _ = surface.load_surf_mesh(fsaverage['pial_%s' % hemi])
    #     for idx, label in enumerate(labels):
    #         #if ("unknown" not in str(label) and "corpuscallosum" not in str(label)):  # Omit the unknown and corpuscallosum labels.
    #         if ('Brodmann.' in str(label)):   # Compute mean location of vertices in label of index idx
    #             #pdb.set_trace()
    #             if hemi=='left':
    #                 label_coord[str(label).split(sep="'")[1]+'-lh'] = np.mean(coords[vert == idx], axis=0)
    #             else:
    #                 label_coord[str(label).split(sep="'")[1]+'-rh'] = np.mean(coords[vert == idx], axis=0)
                
    return label_coord

##############################################################################
def getCoordinates(atlas):
    """
    Get the coordinates corresponding to the Deskian-Killiany's brain areas.

    Returns
    -------
    np.array with the 3D coordinates.

    """
    
    coordinates = []
    
    if atlas == 'DKEgill':
        # Same labels as for connectivity features 
        label_coord = getDKlabelCoordinates()
        # Sort labels according to connectivity featurers
        labels = getLabels(atlas)
        for i in labels:
            coordinates.append(label_coord[i])
    else:
        label_coord = getBAlabelCoordinates()
        
        # Sort labels according to connectivity featurers
        atlas = 'BAita' #Same for BAita and BAitaSig
        labels = getLabels(atlas)
        for i in labels:
            coordinates.append(np.append(label_coord[i], 1))
        
        MNI = np.array([[0.9975, -0.0073, 0.0176, -0.0429], 
                  [0.0146, 1.0009, -0.0024, 1.5496],
                  [-0.0130, -0.0093, 0.9971, 1.1840]])
    
        coordinates = np.array(coordinates)
        coordinates = np.transpose(MNI.dot(np.transpose(coordinates)))
    
    
    
    
    
    return np.array(coordinates)  # 3D coordinates of parcels

##############################################################################
def getPermResults(dir_auc, dir_auc_perm, dir_save, wanted_info, nb_perms, title, con_type):
    """
    Gets the permutation results and prints it as a histogram, and prints the 
    p-value. 

    Parameters
    ----------
    dir_auc : string
        Directory path to where the auc-scores are saved.
    dir_auc_perm : string
        Directory path to where the permutation auc-scores are saved.
    dir_save : string
        Directory path to where the figures should be saved.
    wanted_info : string
        Can be 'both', 'all', or 'bands'
    clf_types : list of string
        The strings tells what classifier to use.

    """
    
    auc_bands_dict = getData(dir_auc, wanted_info, ['lasso'])
    perm_auc_bands_dict = getData(dir_auc_perm, wanted_info, ['lasso'])
    
    
    
    sns.set(font_scale=2)
    fig, ax = plt.subplots(2,3, figsize = (10,7))
    pval_list = []
    for i, band in enumerate(list(auc_bands_dict.keys())) :    
        mu_result = np.mean(auc_bands_dict[band])
        
        result_perm_all = perm_auc_bands_dict[band]
        n_loops = len(result_perm_all)//nb_perms
        
        result_perm = np.mean(np.reshape(result_perm_all[0:(len(result_perm_all)//n_loops)*n_loops], (len(result_perm_all)//n_loops, n_loops)), axis = 1)
        #mu_result_perm_all = np.mean(result_perm_all)
        mu_result_perm = np.mean(result_perm)
        
        #Plot
        sns.distplot(result_perm, bins=20, ax = ax[i//3][i%3], kde = False, color='b', 
                     hist_kws = {'alpha':1})
        ax[i//3][i%3].axvline(mu_result, color='k', linestyle='dashed', linewidth = 3.5)
        ax[i//3][i%3].axvline(mu_result_perm, color='r', linestyle='dashed', linewidth = 3.5)
        ax[i//3][i%3].set_title(band, fontsize= 26)
        ax[i//3][i%3].set_xlabel("Mean AUC's" , fontsize= 23)
        ax[i//3][i%3].set_ylabel('Frequency', fontsize= 23)
        #ax[i//3][i%3].text(0,0,'Frequency')#, fontsize= 23)
        # ax[i//3][i%3].set_ylim([0, 18])
        
        pval = (sum(result_perm>mu_result)+1)/(len(result_perm)+1)
        pval_list.append(pval)
        print(band, "\nNumber of permutations:", len(result_perm), "\nPermutation mean:", mu_result_perm, 
              "\nTrue mean:", mu_result, "\np-value =", np.round(pval,3), "\n")
    
    plt.suptitle(title, fontsize= 30)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.78, bottom = 0.1, hspace = 0.8, wspace = 0.6)
    fig.legend(['Original mean', 'Permutation mean'], bbox_to_anchor = (0.5, 0.94), 
               borderaxespad = 0., loc = 'upper center', ncol = 2)
    
    plt.show()
    
    #pdb.set_trace()
    
    fig.savefig(dir_save + '/PermutationHist_' + con_type + '_' + wanted_info + '.jpg', bbox_inches = 'tight')
    sns.reset_orig()
    return pval_list

##############################################################################
from matplotlib import cm
def plotSigBrainConnections(cons_Wmu, cons_Wcount, atlas, band_idx, dir_save, con_type, denom):
    """
    Plots the connections between different brain parts for the different 
    frequency bands on an image of the brain. The thickness of the connections 
    depends on how the weights have been calculated. Choose a title to explain 
    what weights that are used. 

    Parameters
    ----------
    cons_W_tot : list of lists
        Contains lists corresponding to each frequency band with the weights 
        between each used brain area.
    title : string
        The title you want for the figure.

    """
    
    # Get the coordinates for each brain area
    coordinates = getCoordinates(atlas)
    
    if atlas == 'BAitaSig':
        title = 'DL-sig'
    elif atlas == 'BAita':
        title = 'DL'
    else: 
        title = 'DK-expert'
    
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    title += ' ' + freq_bands[band_idx] +": Connections on brain - " + con_type.upper() 
    #pdb.set_trace()
    #Make symmetric
    
    fig, ax = plt.subplots(1,2, figsize = (12, 7.5))
    fig.subplots_adjust(bottom = 0.32)
    
    if np.min(cons_Wmu)>=0:
        cmap1 = mpl.cm.Reds
        norm1 = mpl.colors.Normalize(vmin=0, vmax=np.max(cons_Wmu))
    elif np.max(cons_Wmu)<=0:
        cmap1 = mpl.cm.Blues.reversed()
        norm1 = mpl.colors.Normalize(vmin=np.max(cons_Wmu), vmax=0)
    elif np.abs(np.min(cons_Wmu))<np.max(cons_Wmu):
        cmap1 = mpl.cm.seismic
        norm1 = mpl.colors.Normalize(vmin=-np.max(cons_Wmu), vmax=np.max(cons_Wmu))
    else:
        cmap1 = mpl.cm.seismic
        norm1 = mpl.colors.Normalize(vmin=-np.min(cons_Wmu), vmax=np.min(cons_Wmu))
    
    fig1 = plotting.plot_connectome(cons_Wmu, coordinates, display_mode='z', axes = ax[0], 
                                    annotate = True, edge_cmap = cmap1)

    plt.title('Weight: average coefficient', y = 1, fontsize= 20)
    fig1.annotate(size = 18)
    
    cmap2 = mpl.cm.Reds
    norm2 = mpl.colors.Normalize(vmin=0, vmax=np.max(cons_Wcount)/denom*100)
    fig2 = plotting.plot_connectome(cons_Wcount, coordinates, display_mode='z', axes = ax[1],
                                    edge_cmap = cmap2)
       
    plt.title('Weight: Nb. of occurrences', y = 1, fontsize= 20)
    fig2.annotate(size = 18)

    fig.suptitle(title, fontsize=25)
    
    marker_color = list(cm.Paired(np.linspace(0,1, (len(coordinates)//2-1))))
    marker_color.extend([np.array([0,0,0,1])])
    
    labels = getLabels(atlas) 
    #pdb.set_trace()
    for i,j in enumerate(range(0, len(coordinates), 2)):
        fig1.add_markers(marker_coords = [coordinates[j], coordinates[j+1]], 
                          marker_size = 200, marker_color = marker_color[i], 
                          label= labels[j].split('-')[0])
        fig2.add_markers(marker_coords = [coordinates[j], coordinates[j+1]], 
                          marker_size = 200, marker_color = marker_color[i], 
                          label= labels[j].split('-')[0])

    plt.legend(bbox_to_anchor = (1.05,1), borderaxespad = 0., loc = 'upper left',
                prop= {'size': 18})
    
    # Colorbar 1
    cbax = fig.add_axes([0.19, 0.29, 0.23, 0.03])
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm1, cmap = cmap1), ax = ax[0], 
                 orientation = 'horizontal', shrink = 0.5, cax = cbax)
    
    # Colorbar 2
    cbax = fig.add_axes([0.62, 0.29, 0.23, 0.03])
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm2, cmap = cmap2), ax = ax[1], 
                 orientation = 'horizontal', shrink = 0.5, cax=cbax)
    
    plt.show()
    
    #pdb.set_trace()
    fig.savefig(dir_save + '/BrainConnections' + con_type + '_' + freq_bands[band_idx] + '.jpg', bbox_inches = 'tight')

##############################################################################
def getItalianSigMatrices():
    #dat = pd.read_csv(dir_y_ID, header = 0, names = ['Id', 'Age', 'Gender', 'Group'])
    dir1 = r'/share/FannyMaster/PythonNew/Lorenzo_SI_table1.xlsx'
    xls = pd.ExcelFile(dir1)
    itaWeights = []

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
        itaWeights.append(dat[roi_mat].fillna(0))

    return itaWeights

##############################################################################
def plotItalianBrainConnections(itaW, band_idx, dir_save):
    """
    Plots the connections between different brain parts for the different 
    frequency bands on an image of the brain. The thickness of the connections 
    depends on how the weights have been calculated. Choose a title to explain 
    what weights that are used. 

    Parameters
    ----------
    cons_W_tot : list of lists
        Contains lists corresponding to each frequency band with the weights 
        between each used brain area.
    title : string
        The title you want for the figure.

    """
    
    # Get the coordinates for each brain area
    coordinates = getCoordinates('BAita')
    
    
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    title = 'Di Lorenzo et al.: ' + freq_bands[band_idx] 
    #pdb.set_trace()
    #Make symmetric
    
    fig ,ax= plt.subplots(1,1 ,figsize = (5, 10))
    fig.subplots_adjust(bottom = 0.5)
    
    itaW_nz = itaW[np.nonzero(itaW)]
    
    if np.min(itaW_nz)>=0:
        cmap1 = mpl.cm.Reds
        norm1 = mpl.colors.Normalize(vmin=np.min(itaW_nz), vmax=np.max(itaW_nz))
    elif np.max(itaW_nz)<=0:
        cmap1 = mpl.cm.Blues.reversed()
        norm1 = mpl.colors.Normalize(vmin=np.min(itaW_nz), vmax=np.max(itaW_nz))
    elif np.abs(np.min(itaW_nz))<np.max(itaW_nz):
        cmap1 = mpl.cm.seismic
        norm1 = mpl.colors.Normalize(vmin=-np.max(itaW_nz), vmax=np.max(itaW_nz))
    else:
        cmap1 = mpl.cm.seismic
        norm1 = mpl.colors.Normalize(vmin=-np.min(itaW_nz), vmax=np.min(itaW_nz))
    
    fig1 = plotting.plot_connectome(itaW, coordinates, display_mode='z', axes = ax, 
                                    annotate = True, edge_cmap = cmap1)
    
    plt.title('Weight: z-value', y = 1, fontsize= 20)
    fig1.annotate(size = 18)
    fig.suptitle(title, fontsize=25)
    
    marker_color = list(cm.Paired(np.linspace(0,1, (len(coordinates)//2-1))))
    marker_color.extend([np.array([0,0,0,1])])
    
    labels = getLabels('BAita') 
    #pdb.set_trace()
    for i,j in enumerate(range(0, len(coordinates), 2)):
        fig1.add_markers(marker_coords = [coordinates[j], coordinates[j+1]], 
                          marker_size = 200, marker_color = marker_color[i], 
                          label= labels[j].split('-')[0])

    plt.legend(bbox_to_anchor = (1.05,1.15), borderaxespad = 0., loc = 'upper left',
                prop= {'size': 18}, ncol= 1)
    
    # Colorbar 1
    cbax = fig.add_axes([0.3, 0.47, 0.45, 0.02])
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm1, cmap = cmap1), ax = ax, 
                 orientation = 'horizontal', shrink = 0.5, cax = cbax) #pad=0.2,
    
    plt.show()
    
    #pdb.set_trace()
    fig.savefig(dir_save + '/BrainConnections_Italians_' + freq_bands[band_idx] + '.jpg', bbox_inches = 'tight')


##############################################################################
def plotItalianBrainConnections2(itaW, itaW2, band_idx, dir_save):
    """
    Plots the connections between different brain parts for the different 
    frequency bands on an image of the brain. The thickness of the connections 
    depends on how the weights have been calculated. Choose a title to explain 
    what weights that are used. 

    Parameters
    ----------
    cons_W_tot : list of lists
        Contains lists corresponding to each frequency band with the weights 
        between each used brain area.
    title : string
        The title you want for the figure.

    """
    
    # Get the coordinates for each brain area
    coordinates = getCoordinates('BAita')
    #pdb.set_trace()
    freq_bands = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    title = 'Di Lorenzo et al.: ' + freq_bands[band_idx]
    #pdb.set_trace()
    #Make symmetric
    
    fig, ax = plt.subplots(1,2, figsize = (10, 7.5))
    fig.subplots_adjust(bottom = 0.32)
    
    itaW_nz = itaW[np.nonzero(itaW)]
    
    if np.min(itaW_nz)>=0:
        cmap1 = mpl.cm.Reds
        norm1 = mpl.colors.Normalize(vmin=np.min(itaW_nz), vmax=np.max(itaW_nz))
    elif np.max(itaW_nz)<=0:
        cmap1 = mpl.cm.Blues.reversed()
        norm1 = mpl.colors.Normalize(vmin=np.min(itaW_nz), vmax=np.max(itaW_nz))
    elif np.abs(np.min(itaW_nz))<np.max(itaW_nz):
        cmap1 = mpl.cm.seismic
        norm1 = mpl.colors.Normalize(vmin=-np.max(itaW_nz), vmax=np.max(itaW_nz))
    else:
        cmap1 = mpl.cm.seismic
        norm1 = mpl.colors.Normalize(vmin=-np.min(itaW_nz), vmax=np.min(itaW_nz))
    
    fig1 = plotting.plot_connectome(itaW, coordinates, display_mode='z', axes = ax[0], 
                                    annotate = True, edge_cmap = cmap1)
    
    plt.title('Weight: z-value', y = 1, fontsize= 20)
    fig1.annotate(size = 18)
    
    fig2 = plotting.plot_connectome(itaW2, coordinates, display_mode='z', axes = ax[1],
                                    edge_cmap = cmap1)  
    plt.title('Top 7 z-values', y = 1, fontsize= 20)
    fig2.annotate(size = 18)

    fig.suptitle(title, fontsize=25)
    
    marker_color = list(cm.Paired(np.linspace(0,1, (len(coordinates)//2-1))))
    marker_color.extend([np.array([0,0,0,1])])
    
    labels = getLabels('BAita') 
    #pdb.set_trace()
    for i,j in enumerate(range(0, len(coordinates), 2)):
        fig1.add_markers(marker_coords = [coordinates[j], coordinates[j+1]], 
                          marker_size = 200, marker_color = marker_color[i], 
                          label= labels[j].split('-')[0])
        fig2.add_markers(marker_coords = [coordinates[j], coordinates[j+1]], 
                          marker_size = 200, marker_color = marker_color[i], 
                          label= labels[j].split('-')[0])
  
    plt.legend(bbox_to_anchor = (1.05,1), borderaxespad = 0., loc = 'upper left',
                    prop= {'size': 18})
    
    # Colorbar 1
    cbax = fig.add_axes([0.19, 0.29, 0.23, 0.03])
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm1, cmap = cmap1), ax = ax[0], 
                 orientation = 'horizontal', shrink = 0.5, cax = cbax)
    
    # Colorbar 2
    cbax = fig.add_axes([0.62, 0.29, 0.23, 0.03])
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm1, cmap = cmap1), ax = ax[1], 
                  orientation = 'horizontal', shrink = 0.5, cax=cbax)
    plt.show()
    
    fig.savefig(dir_save + '/BrainConnections_Top7' + freq_bands[band_idx] + '.jpg', bbox_inches = 'tight')

##############################################################################

