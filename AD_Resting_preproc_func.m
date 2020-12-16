function [ outEEG ] = AD_Resting_preproc_func( inEEG , desiredFs, lowCut, highCut)

%
% function outEEG = MdDS_AEP_preproc_func( inEEG , desiredFs, lowCut, highCut)
%
%  1. downsampling based on desiredFs
%  2. band-pass filtering based on lowPass & highPass
%  3. artifact subspace reconstruction (ASR)
%  4. channel interpolation 
%  5. re-referencing
%
% *** INPUT ***
%   EEG : eeglab structure
%   desiredFs [scalar] : sampling rate (Hz) to be downsampled
%   lowCut [scalar] : lowCut frequency (Hz) : high-pass filter
%   highCut [scalar] : highCut frequency (Hz) : low-pass filter
%
% *** OUTPUT ***
%   outEEG : eeglab structure after preprocessing
%
% *** USEAGE ***
%
% outEEG = GLAD_RSEEG_preproc_func( inEEG, desiredFs, lowCut, highCut )
%
% Frohlich Lab. Sangtae Ahn (sangtae_ahn@med.unc.edu)
% 
% first written by 11/30/2016
% revised by 12/13/2016 : change parameters as GUI default setting in clean_rawdata
% revised by 1/10/2017 : change WindowCriterion 
% revised by 2/15/2017 : change parameters 
%

%% Resampling and band-pass filtering
fc=0.9;
df=0.2;

disp(['band-pass filtering from '  num2str(lowCut) ' to ' num2str(highCut)  ' Hz']);
EEG = pop_eegfiltnew(inEEG, lowCut, highCut);


disp(['resampled with : ' num2str(EEG.srate) 'Hz']);
EEG = pop_resample( EEG, desiredFs,fc,df);


EEG = eeg_checkset( EEG );




%% Save chanloc structure for future use (Interpolatation)
EEG.etc.historychanlocs=EEG.chanlocs; % Save channel locs of 128 because ASR will remove bad channel loc information
EEG.etc.historychaninfo=EEG.chaninfo;

%% Run Artifact subspace reconstruction (removes bad epoch data (PCA), bad channels)
EEG = clean_rawdata(EEG,5,-1,0.8,4,5,-1); % default setting
EEG.etc.badchan=find(EEG.etc.clean_channel_mask==0); %Bad chananel information from ASR
EEG = eeg_checkset( EEG );

%% Interpolate bad channels
EEG.etc.originalEEG=EEG; % keep origianl EEG before interpolation
EEG = pop_interp(EEG, EEG.etc.historychanlocs, 'spherical');

%% Rereference to AVREF (should be with interpolated channels included!)
EEG = eeg_checkset( EEG );
EEG = pop_reref( EEG, []);

outEEG=EEG;

end

