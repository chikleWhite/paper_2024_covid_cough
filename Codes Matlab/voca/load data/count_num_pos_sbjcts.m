%% load metadata + positive subjects metadata

% remove previous data
close all; clc; clear;

% load metadata from database
prev_folder = cd('..\..\..\');
folder_path = pwd;
folder_name = folder_path + "\database\voca\voca-dataset-09-09-20";
md = jsondecode(fileread(folder_name + "\submissions.json")); % metadata
cd(prev_folder);

% pre-allocation
num_subj = length(md);
pos_count = 0;

for i = 1 : num_subj % loop over all subjects in metadata
    
    md_subj = md{i}; % metadata of 1 subject
    
    % check requirements
    if      ~isfield(md_subj.formData, 'covid19')
        continue;
        
    % collect subject info
    else
        covid_status    = string(md_subj.formData.covid19.diagnosedCovid19);
%         PCR_test_date   = md_subj.formData.covid19.testDate;
%         test_symptoms   = md_subj.formData.covid19.testSymptoms;
%         recording_date  = md_subj.datetime;
%         age             = md_subj.formData.age;
%         gender          = md_subj.formData.gender;
%         current_symptoms = md_subj.formData.covid19.currentSymptoms;
    end
    
    if covid_status == "Yes" % subject is positive
        pos_count = pos_count + 1;
    end
end