%% load metadata and get list of background diseases

% remove previous data
close all; clc; clear;

% load metadata from database
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "\database\voca\voca-dataset-09-09-20";
metaData = jsondecode(fileread(folder_name + "\submissions.json")); % metadata
cd(prev_folder);

% pre-allocation
numSubj = length(metaData);
backgroundDiseases = cell(numSubj, 1);
backgroundDiseasesTypes = [];

% loop over all subjects
for row = 1 : numSubj % loop over all subjects in metadata
    
    metaDataSubj = metaData{row}; % metadata of 1 subject
    
    % check requirements
    if      ~isfield(metaDataSubj.formData, 'covid19') || ...
            ~exist(folder_name + "\" + metaDataSubj.x_id + "\cough.wav", 'file') || ...
            ~isfield(metaDataSubj.formData, 'backgroundDiseases') || ...
            isempty(metaDataSubj.formData.backgroundDiseases)
        continue;
        
    % collect subject info
    else
        backgroundDiseases{row} = metaDataSubj.formData.backgroundDiseases;
    end
end

% loop over all backgroundDiseases
for row = 1 : numSubj
    
    backgroundDiseasesTypesPerRow = string(backgroundDiseases{row});
    numDiseases = length(backgroundDiseasesTypesPerRow);
    
    % loop over all diseases per subject
    for diseasesNum = 1 : numDiseases
        if sum(strcmp(backgroundDiseasesTypes, backgroundDiseasesTypesPerRow(diseasesNum))) == 0
            backgroundDiseasesTypes = ...
                cat(1, backgroundDiseasesTypes, backgroundDiseasesTypesPerRow(diseasesNum));
        end
        
    end
    
end
