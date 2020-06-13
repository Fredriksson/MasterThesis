# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:28:02 2020

@author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
"""
from os import chdir
chdir(r'/share/FannyMaster/PythonNew')

import pandas as pd #Dataframes
from utilsMakeTable1 import make_dat, make_summary

#%%##################################################################
# Get pandas dataframe with wanted features
#####################################################################
dir1 =  r'/share/FannyMaster/PythonNew/pecans_bl_and_panss6w_meds6w.xlsx'
dir2 =  r'/share/FannyMaster/PythonNew/Cognition_Pecans1.xls'
dat = make_dat(dir1, dir2)

#%%##################################################################
# Make Table 1
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6877469/pdf/S0033291718003781a.pdf
#####################################################################
summary = make_summary(dat)

#%%
# Print in latex format
dfSummary = pd.DataFrame(summary).transpose()
dfSummary.columns = ['$N$', 'Mean (SD)', '$N$', 'Mean (SD)']
print(dfSummary.to_latex())

#%% Save values
age_gender = dat[['Id', 'Age [years]', 'Gender']]

age_gender.to_csv(r'C:\Users\FFRE0009\Documents\PythonNew\Age_Gender.csv', index = False)












