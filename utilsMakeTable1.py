# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:38:50 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""


import pandas as pd #Dataframes
import numpy as np
from os import listdir

#%%##################################################################
# Make excel file containing wanted information
#####################################################################

def make_dat(dir1, dir2):
    # Extract the id's of the used subjects
    used_id = listdir('/data/EEGdata/PECANS1/RSA')
    used_id = np.unique([i.split('R')[0].upper() for i in used_id])
    
    # First excel file
    #dir1 =  r'P:\PC Glostrup\Lukkede Mapper\Forskningsenhed\CNSR\Fanny\pecans_bl_and_panss6w_meds6w.xlsx'
    xls = pd.ExcelFile(dir1)
    
    demo = make_demo(xls, used_id);
    
    # Note what patients that are schizophrenic
    hc = demo['Id'][demo['Group']==0].reset_index(drop=True)
    used_scz = set(used_id) - set(hc)
    
    subst_life = make_subst_life(xls, used_id)
    func = make_func(xls, used_scz)
    panss = make_panss(xls, used_scz)
    psych = make_psych(xls, used_scz)
    
    # Second excel file
    #dir2 =  r'P:\PC Glostrup\Lukkede Mapper\Forskningsenhed\CNSR\Fanny\Cognition_Pecans1.xls'
    xls = pd.ExcelFile(dir2)
    cognition = make_cognition(xls, used_id)
    
    #dat = pd.concat([demo, subst_life, cognition, func, panss, psych], axis = 1)
    
    dat = demo.merge(subst_life, left_on = 'Id', right_on= 'Id', how = 'outer')
    dat = dat.merge(cognition, left_on = 'Id', right_on= 'Id', how = 'outer')
    dat = dat.merge(func, left_on = 'Id', right_on= 'Id', how = 'outer')
    dat = dat.merge(panss, left_on = 'Id', right_on= 'Id', how = 'outer')
    dat = dat.merge(psych, left_on = 'Id', right_on= 'Id', how = 'outer')
    return dat



#%% Process the demo sheet

def make_demo(xls, used_id):
        
    #Age at scan    
    demo = pd.read_excel(xls, 'demo', na_values=['-8', '-9', '-888'])
    
    temp = demo['Date of inclusion'] - demo['Birthday']
    demo.insert(3, "Age [years]", temp.dt.days/365, True)
    
    # Save only wanted columns
    demo = demo[['Id', 'Age [years]', 'Gender', 'Group', 'Hand', 'Hand_score', 
                 'Parents Socioeconomic Status', 'Sub_edyr']]
    demo.columns = ['Id', 'Age [years]', 'Gender', 'Group', 'Hand', 'Hand_score', 
                 'Parental SES', 'Sub_edyr']
    # Make Sub_edyr numeric
    demo.loc[:, 'Sub_edyr'] = pd.to_numeric(demo['Sub_edyr'], errors='coerce')
    
    # Make Id to upper cases
    demo['Id'] = [i.upper() for i in demo['Id']]
    demo['Parental SES'] = [i.lower() if type(i) == str else i for i in demo['Parental SES']]
    
    # Make Parents Socioeconomic Status numeric
    # 0: A, 1: B, 2: C
    #demo['Parental SES'].replace({'A': 0, 'a': 0, 'B': 1, 'b': 1, 'C': 2, 'c':2},inplace = True)
    #demo.loc[:, 'Parental SES'] = pd.to_numeric(demo['Parental SES'], errors='coerce')
        
    # Check that all the used subjects are represented in demo
    if (set(used_id)- set(demo['Id'])) != set():
        print('Missing subjects', set(used_id)- set(demo['Id']), 
              'in demo.')
    
    # Drop unused subjects
    use = demo['Id'].isin(used_id);
    demo = demo[use].reset_index(drop=True);
    
    # Return demo
    return demo


#%% #%% Process the subst_life sheet (Benz= sleep medicin, Opioids = stimulating (opium))
def make_subst_life(xls, used_id):
    subst_life = pd.read_excel(xls, 'subst_life', na_values=['-8', '-9'])
    # Save only wanted columns
    subst_life = subst_life[['Id', 'Coffee', 'Tea', 'Other_Caf', 'Alc', 'Tobacco', 'Cannabis', 
                            'Benz', 'Opioids', 'Stimulants', 'Hallusinogenes', 'Other drugs']]
    # Make Id to upper cases
    subst_life['Id'] = [i.upper() for i in subst_life['Id']]
    
    # Check that all the used subjects are represented in subst_life
    if (set(used_id)- set(subst_life['Id'])) != set():
        print('Missing subjects', set(used_id)- set(subst_life['Id']), 
              'in subst_life.')
    
    # Drop unused subjects
    use = subst_life['Id'].isin(used_id);
    subst_life = subst_life[use].reset_index(drop=True);
    
    # Return subst_life
    return subst_life

#%% #%% Process the Func sheet
# CGI_sev1 (clinical global empression scale - severity)
# GAF_S (Global assesment of Functional - symptoms)
# GAF_F (Global assesment of Functional - function)
# SOFAS_A

def make_func(xls, used_scz):
    
    func = pd.read_excel(xls, 'Func', na_values=['-8', '-9'])
    
    # Save only wanted columns
    func = func[['Id', 'CGI_sev1', 'GAF_S', 'GAF_F', 'SOFAS_A']]
    # Make Id to upper cases
    func['Id'] = [i.upper() for i in func['Id']]
    
    # Check that all the used subjects are represented in func
    if (set(used_scz)- set(func['Id'])) != set():
        print('Missing subjects', set(used_scz)- set(func['Id']), 
              'in func.')
 
    # Drop unused subjects
    use = func['Id'].isin(used_scz);
    func = func[use].reset_index(drop=True);

    # Retrun func
    return func


#%% PANSS_BL_6w
# sum p, n, g, tot

def make_panss(xls, used_scz):
    panss = pd.read_excel(xls, 'PANSS_BL_6w')
    
    # Only use visitType 0 (=first visit)
    panss = panss[panss['visitType']==0].reset_index(drop=True)
    
    # Get total p values
    p = panss[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']]
    p = p.sum(axis=1)
    panss.insert(1, "PANSS positive", p, True)
    
    # Get total n values
    n = panss[['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']]
    n = n.sum(axis=1)
    panss.insert(2, "PANSS negative", n, True)
    
    # Get total g values
    g = panss[['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 
               'g12', 'g13', 'g14', 'g15', 'g16']]
    g = g.sum(axis=1)
    panss.insert(3, "PANSS general", g, True)
    
    # Get total values
    tot = p+n+g;
    panss.insert(4, "PANSS total", tot, True)
    
    # Save only wanted columns
    panss = panss[['Id', 'PANSS positive', 'PANSS negative', 'PANSS general', 'PANSS total']]
    
    # Check that all the used subjects are represented in panss
    if (set(used_scz)- set(panss['Id'])) != set():
        print('Missing subjects', set(used_scz)- set(panss['Id']), 
              'in panss.')
    
    # Drop unused subjects
    use = panss['Id'].isin(used_scz);
    panss = panss[use].reset_index(drop=True);
    
    # Return panss
    return panss

#%% psych
#DUI (duration of untreated illness) [weeks]
def make_psych(xls, used_scz):
    psych = pd.read_excel(xls, 'psych', na_values=['-8', '-9'])
    
    # Save only wanted 
    psych = psych[['Id', 'DUI']]
    
    # Check that all the used subjects are represented in panss
    if (set(used_scz)- set(psych['Id'])) != set():
        print('Missing subjects', set(used_scz)- set(psych['Id']), 
              'in psych.')
    
    # Drop unused subjects
    use = psych['Id'].isin(used_scz);
    psych = psych[use].reset_index(drop=True);
    
    return psych


#%% Cognition
# Dart (middel + std)

# WAIS-lll (tag gennemsnit) 
# 1 middel av de 4 værdier
# 2. Middel af alle syge og alle raske
# 3. Raske: (middel(raske)-middel(raske))/middel(raske) = 0   -> middel = 0, std = 1
# 4.z-score(syge) = (middel(syge)-middel(raske))/std(raske)) -> forvente middel < 0 (Karen -0.94 std 1.5)

def make_cognition(xls, used_id):
    cognition = pd.read_excel(xls, 'Sheet1', skiprows = [0,2], header = 0)    
    # Save mean of WAIS-III as column
    wais = cognition[['WAIS-III_Block_design', 'WAIS-III_Matrix_reasoning', 'WAIS-III_Similarities', 'WAIS-III_Vocabulary']]
    wais = wais.mean(axis=1)
    cognition.insert(1, "WAIS-III", wais, True)
    
    # Save only wanted columns
    cognition = cognition[['Unnamed: 0', 'DART', 'WAIS-III']]
    cognition.columns = ['Id', 'DART', 'WAIS-III']
    
    # Check that all the used subjects are represented in cognition
    if (set(used_id)- set(cognition['Id'])) != set():
        print('Missing subjects', set(used_id)- set(cognition['Id']), 
              'in cognition.')
    
    # Drop unused subjects
    use = cognition['Id'].isin(used_id);
    cognition = cognition[use].reset_index(drop=True);

    # Return cognition
    return cognition


#%%########################################################################
######## Make summary
###########################################################################

def make_summary(dat):
    dat_scz = dat[dat['Group']==1].reset_index(drop=True)
    dat_hc = dat[dat['Group']==0].reset_index(drop=True)
    
    summary = {}
    summary['Number of patients'] = [dat_scz.shape[0],'-', dat_hc.shape[0], '-'];
    
    summary = make_category(dat_scz, dat_hc, summary)
    summary = make_four_cat(dat_scz, dat_hc, summary)
    summary = make_continuous(dat_scz, dat_hc, summary)
    
    return summary
    
    #%%
def make_category(dat_scz, dat_hc, summary):
    ind = ['Gender', 'Hand', 'Parental SES', 'Coffee', 'Tea', 'Other_Caf']
    count_scz = dat_scz.count();
    count_hc = dat_hc.count();
    
    # Gender 1=man 0=woman
    for i in ind: 
        keys_scz = ''.join(str(j)+'/' for j in dat_scz[i].value_counts().sort_index())[:-1]
        keys_hc = ''.join(str(j)+'/' for j in dat_hc[i].value_counts().sort_index())[:-1]
        if dat_scz[i].dtypes == float:
            keys_name = ''.join(str(int(j))+'/' for j in dat_scz[i].value_counts().sort_index().keys())[:-1]
        else:
            keys_name = ''.join(j+'/' for j in dat_scz[i].value_counts().sort_index().keys())[:-1]
        summary[i+' ('+keys_name+')'] = [count_scz[i],  keys_scz, count_hc[i], keys_hc]
        return summary
      
    #%%
def make_four_cat(dat_scz, dat_hc, summary):
    ind = ['Alc', 'Tobacco', 'Cannabis', 'Benz', 'Opioids', 'Stimulants', 'Hallusinogenes', 'Other drugs']
    count_scz = dat_scz.count();
    count_hc = dat_hc.count();
    
    for i in ind: 
        
        keys_scz = ''.join(str(j)+'/' for j in dat_scz[i].value_counts().reindex(range(5), fill_value=0))[:-1]
        keys_hc = ''.join(str(j)+'/' for j in dat_hc[i].value_counts().reindex(range(5), fill_value=0))[:-1]
        
        keys_name = ''.join(str(int(j))+'/' for j in dat_scz[i].value_counts().reindex(range(5), fill_value=0).keys())[:-1]
        summary[i+' ('+keys_name+')'] = [count_scz[i],  keys_scz, count_hc[i], keys_hc]
    
    return summary    
    
    #%%
def make_continuous(dat_scz, dat_hc, summary):
    # Calculate mean for all numeric values but the binary
    ind = ['Age [years]', 'Hand_score', 'Sub_edyr', 'DART', 'WAIS-III', 'CGI_sev1',
           'GAF_S', 'GAF_F', 'SOFAS_A', 'PANSS positive', 'PANSS negative',
           'PANSS general', 'PANSS total', 'DUI']
    count_scz = dat_scz.count();
    count_hc = dat_hc.count();
    
    # WAIS : Make z-val
    # 1. middel av de 4 værdier
    # 2. Middel af alle syge og alle raske
    # 3. Raske: (middel(raske)-middel(raske))/sd(raske) = 0   -> middel = 0,  std/std =1 -> std = 1
    # 4. middel(syge) = (middel(syge)-middel(raske))/std(raske)) -> forvente middel < 0 (Karen -0.94 std 1.5)
    # 5. sd(syge) = sd(syge)/sd(raske)
    for i in ind:
        mean_scz = round(dat_scz[i].mean(),1)
        sd_scz = round(dat_scz[i].std(),1)
        mean_hc = round(dat_hc[i].mean(),1)
        sd_hc = round(dat_hc[i].std(),1)
        
        if np.isnan(mean_hc):
            summary[i] = [count_scz[i], str(mean_scz) +' (' + str(sd_scz) + ')', 
                          '-', '-']
            
        elif i != "WAIS-III": 
            summary[i] = [count_scz[i], str(mean_scz) +' (' + str(sd_scz) + ')', 
                          count_hc[i], str(mean_hc) + ' (' + str(sd_hc)+')']
        else: 
            z_mean = str(round((mean_scz-mean_hc)/sd_hc,1))
            z_sd = str(round(sd_scz/sd_hc,1))
            summary['Total IQ (WAIS III)']  = [count_scz[i], z_mean + ' ('+ z_sd +')' 
                                      , count_hc[i], '0 (1)']
    return summary

















