

%
%
% AD preprocessing
%
% Sangtae Ahn (stahn@knu.ac.kr)
%
%
% first written 8/1/2020
%
%



close all
clear
clc


%% Load dataset

addpath('D:\OneDrive - knu.ac.kr\Matlab\toolbox\eeglab2019_1');
addpath('D:\OneDrive - knu.ac.kr\Matlab\AD');
eeglab;
pop_editoptions( 'option_savetwofiles', 1,'option_single', 0);

dPath='D:\Matlab\Data\AD\';
cd(dPath);
subStruct = dir;
subStruct = subStruct(cellfun(@any,strfind({subStruct.name},'sub')));
nSub=length(subStruct);

lowCut=1;
highCut=50;
desiredFs=500;

%%

for iSub = 1
    
    eeglab
    
    load(subStruct(iSub).name);
    
    subId=eeg.id;
    EEG.srate=eeg(1).resting.srate;
    EEG.data=eeg(1).resting.raw_data_eo;
    EEG.chanlocs=eeg(1).resting.locs;
    eeglab redraw
    pop_saveset(EEG,'filepath',[dPath 'preproc\'],'filename',[subId '_AD_Resting.set']);
    
    EEG = AD_Resting_preproc_func(EEG,desiredFs,lowCut,highCut);
    pop_saveset(EEG,'filepath',[dPath 'preproc\'],'filename',[subId '_AD_Resting_p.set']);
    
    EEG.rank=rank(double(EEG.data));
    EEG = pop_runica(EEG,'extended',1,'pca',EEG.rank);
    pop_saveset(EEG,'filepath',[dPath 'preproc\'],'filename',[subId '_AD_Resting_pi.set']);
    
    
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
    
    pop_saveset(EEG,'filepath',[dPath 'preproc\'],'filename',[subId '_AD_Resting_pir.set']);
    
    clear EEG;
    
end







