

%
%
% AD MMSE
%
% Sangtae Ahn (stahn@knu.ac.kr)
%
% load all data and extract MMSE scores
% 
% first written 9/29/2020
%
%



close all
clear
clc


%% Load dataset

MAIN_PATH = 'D:\OneDrive - knu.ac.kr\Matlab\AD';

dPath=[MAIN_PATH '\Data\'];
cd(dPath);
subStruct = dir;
subStruct = subStruct(cellfun(@any,strfind({subStruct.name},'sub')));
nSub=length(subStruct);


%%

for iSub = 1 : nSub
    
    load(subStruct(iSub).name);
    
%     subId=eeg.id;

    MMSE = [];
%     eeg(1).information
    MMSE = [MMSE eeg(1).information.MMSE_tot];
 
    
    
    
end







