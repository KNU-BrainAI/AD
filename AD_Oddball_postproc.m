

%
%
% AD postprocessing for Visual Oddball data
%
% Sangtae Ahn (stahn@knu.ac.kr)
%
%
% first written 12/23/2020
%
%



close all
clear
clc


%% Load dataset

addpath('D:\OneDrive - knu.ac.kr\Matlab\toolbox\eeglab2019_1'); % add eeglab path
addpath('D:\OneDrive - knu.ac.kr\Matlab\AD'); % add code path
eeglab; % run eeglab
pop_editoptions( 'option_savetwofiles', 1,'option_single', 0); % save data into two files: .set and .fdt

dPath='D:\Matlab\Data\AD\preproc\oddball\'; % data path
cd(dPath);
fileStruct = dir;
fileStruct = fileStruct(cellfun(@any,strfind({fileStruct.name},'_p.set')));
nSub=length(fileStruct);


%% MAIN LOOP

for iSub = 1 : nSub
    
    fileId = fileStruct(iSub).name;
    
    disp(['load file... ' fileId]);
    EEG = pop_loadset(fileId); 
    
    % epoching
    tmin = -0.2; % in second
    tmax = 0.6; % in second
    EEG = pop_epoch( EEG, {'dev' 'std'}, [tmin tmax]); 
    
    % baseline correction
    baseline = [-200 0]; % in millisecond
    EEG = pop_rmbase( EEG, baseline);
    
    % plot
    figure; 
    range = [1 : length(EEG.chanlocs)];
    avg = 0; % plot averaged ERP
    pop_plottopo(EEG, range , 'ERPs', avg); 
    
end

    
    
    
    
    
    
end






