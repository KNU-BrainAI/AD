

%
%
% AD preprocessing for Visual Oddball data
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

dPath='D:\Matlab\Data\AD\'; % data path
cd(dPath);
subStruct = dir;
subStruct = subStruct(cellfun(@any,strfind({subStruct.name},'sub')));
nSub=length(subStruct);

lowCut=1; % low-cut for band-pass filtering
highCut=30; % high-cut for band-pass filtering
desiredFs=500;

%% MAIN LOOP

nSess = 1; % session number


for iSub = 1
    
    eeglab
    
    load(subStruct(iSub).name);
    
    subId=eeg.id; % get sub ID
    EEG.srate=eeg(nSess).oddball.srate; % get sampling rate 
    EEG.data=eeg(nSess).oddball.raw_data;  % get oddball data 
    EEG.chanlocs=eeg(nSess).oddball.locs;
    eeglab redraw % run eeglab to store data into EEG structure
    
    
    % extract events 
    markers={'dev','std'};
    for i = 1 : length(eeg(nSess).oddball.event)
        EEG.event(i).type = markers{eeg(nSess).oddball.event(i).mark};
        EEG.event(i).latency = eeg(nSess).oddball.event(i).onset;
        EEG.event(i).response = eeg(nSess).oddball.event(i).response;
        EEG.event(i).response_time = eeg(nSess).oddball.event(i).response_time;
    end
    EEG = eeg_checkset(EEG, 'eventconsistency'); % Check all events for consistency
    
    pop_saveset(EEG,'filepath',[dPath 'preproc\Oddball\'],'filename',[subId '_AD_Oddball.set']); 
    
    
    % filtering, resampling, ban channel rejection, interpolation, referencing
    EEG = AD_Oddball_preproc_func(EEG,desiredFs,lowCut,highCut);
    pop_saveset(EEG,'filepath',[dPath 'preproc\Oddball\'],'filename',[subId '_AD_Oddball_p.set']);
    
    % run ICA
    EEG.rank=rank(double(EEG.data));
    EEG = pop_runica(EEG,'extended',1,'pca',EEG.rank);
    pop_saveset(EEG,'filepath',[dPath 'preproc\Oddball\'],'filename',[subId '_AD_Oddball_pi.set']);
    
    
    % run ICLabel to find IC components
    EEG = pop_iclabel(EEG, 'default');
    
    rejIdx=[];
    cutProb=0.5; % 50percent
    for iICA = 1 : EEG.rank
        [maxProb maxIdx]= max(EEG.etc.ic_classification.ICLabel.classifications(iICA, :));
        % 1: brain / 2: Muscle / 3: Eye / 4: Heart / 5: Line Noise / 6: Channel Noise / 7: Other
        if maxIdx ~= 1 && maxIdx ~= 7 && maxProb > cutProb
            rejIdx = [rejIdx iICA];
        end
    end
    %        pop_viewprops( EEG , 0 , 1 : EEG.rank);
    EEG.etc.rejIdx = rejIdx;
    EEG = pop_subcomp( EEG, rejIdx, 0);
    
    pop_saveset(EEG,'filepath',[dPath 'preproc\Oddball\'],'filename',[subId '_AD_Oddball_pir.set']);
    
    clear EEG;
    
end






