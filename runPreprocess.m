%% Script description
% Script to preprocess all the given raw EEG data. Data is preprocessed by
% the following: 
%       Resample to 250 Hz per second
%       Bandpass filter (1,100) using Hamming windowed sinc FIR filter
%       Clean data from artifacts by eeglab's clean_artifacts
%       Interpolation of removed channels from clean_artifacts
%       Avergare reference
%       Notch filtering (48,51) to remove power noise (~50Hz)
%       ICA to remove artifacts (eye movements, muscle movements, etc.)
%       Reconstruct ICA with components where brain has the highest prob
%       
% Author: Fanny Fredriksson and Karen Marie Sandø Ambrosen
%%
clc; clear all; close all;

addpath('/home/kmsa/eeglab/eeglab2019_1');
eeglab
close all

%% Parameters
% List of all subjects
ID_list = dir('/data/EEGdata/PECANS1/RSA/D*A.bdf');
%ID_pre = dir('/share/FannyMaster/Preprocessed/*.set');

time = zeros(1, numel(ID_list));
filtorder = []; revfilt = 1;

for sub = 1:numel(ID_list) 
    % Start timer
    tic    
    % Get subject ID
    ID = ID_list(sub).name(1:end-4);

    %% Reading the data in
    EEG = pop_biosig(['/data/EEGdata/PECANS1/RSA/' ID '.bdf'], 'ref',[70 71] ,...
        'refoptions',{'keepref' 'off'});
    EEG = pop_select( EEG,'nochannel',{'S1' 'S2VO' 'VB' 'HR' 'HL' 'Nz'});
    % Add channel locations
    EEG = pop_chanedit(EEG, 'lookup','/home/kmsa/eeglab/eeglab2019_1/plugins/dipfit/standard_BEM/elec/standard_1005.elc');

    %% Preprocessing
    % Resample to 250 Hz per second
    EEGresamp = pop_resample(EEG, 256);

    % Bandpass filter - Filter data using Hamming windowed sinc FIR filter
    EEGbp = pop_eegfiltnew(EEGresamp, 1, 100);

    % Clean data from artifacts
    [EEGclean,~,~] = clean_artifacts(EEGbp, 'WindowCriterion', 0.3, 'Highpass', 'off');

    % Interpolation of removed channels from clean_artifacts
    EEGinter = pop_interp(EEGclean, EEG.chanlocs, 'spherical');

    % Avergare reference
    EEGavgref = pop_reref(EEGinter, []);

    % Notch filtering
    % filtorder = []; revfilt = 1;
    EEGnotch = pop_eegfiltnew(EEGavgref,48,51,filtorder, revfilt);

    % Run ICA
    EEGica = pop_runica(EEGnotch, 'extended',1,'interupt','on');
    EEGlabels = iclabel(EEGica);

    %% Reconstruct 
    % Choose components where brain has the highest probability
    [val, idx] = max(EEGlabels.etc.ic_classification.ICLabel.classifications, [], 2);
    remove = find(idx~=1);
    EEGout = pop_subcomp(EEGica, remove);

    %% End timer
    time(sub) = toc;
    
    %% Save the preprocessed data
    save(['/share/FannyMaster/Preprocessed/' ID '_preprocessed.mat'],'-v7.3')
    pop_saveset(EEGout, 'filename', ['/share/FannyMaster/Preprocessed/' ID '_preprocessed.set']);

end

