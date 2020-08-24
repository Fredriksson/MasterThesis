# MasterThesis
The codes used in the master thesis project "Resting State EEG in Computational Psychiatry".

# Python Scripts
main_runOnce: runs the following scripts:  
      * utils_runOnce_sourceLocalization: Uses preprocessed data (channels x time) and source localize it (brain areas x time).
      * utils_runOnce_connectivity: Uses the source localized data (brain areas x time) and calculates a connectivity matrix. Saves the upper half of the connectivty matrix for                                       each subject ((brain areas x (brain areas-1)) / 2). Also saves the average connectivity matrix and the variance for group 0 and 1 and the 't-                                       statistics' between the groups.
      * utils_runOnce_classification: Uses the connectivity measures ((brain areas x (brain areas-1)) / 2) and try to classify wether subject are group 0 or 1. 


main_resultsConnectivity: plots results regarding the connectivity measures.
      utils_resultsConnectivity: the functions used to plot.


main_resultsClassification: plots and prints results regarding the classification.
      utils_resultsClassificationy: the functions used to plot and print. 
      
utils_joint: functions jointly used across several different scripts.

univariate_ttest: code that makes a univariate t-test between the connectivity matrices of group 0 and 1.

parameter_optimization: code to run to visualze effect of parameter intervals for the classification parameter and thereby optimize it.

QualityCheck: Code that plots preprocessed data to visually check the quality.

understandConnectivityMeasures: Code to visualize and understand different connectivity measures.

# Matlab Scripts
runPreprocess: preprocess the raw EEG data



