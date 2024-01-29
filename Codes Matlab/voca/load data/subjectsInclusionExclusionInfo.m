%%

% remove previous data
close all; clc; clear;

% load metadata from database
prevFolder = cd('..\..\..\..\');
folderPath = pwd;
folderName = folderPath + "\database\voca\voca-dataset-09-09-20";
metadataOrigin = jsondecode(fileread(folderName + "\submissions.json")); % metadata
cd(prevFolder);


% pre-allocation
numSubj = length(metadataOrigin);
posCount = 0;
negCount = 0;

for i = 1 : numSubj % loop over all subjects in metadata
    
    mdSubj = metadataOrigin{i}; % metadata of 1 subject
    
    % check requirements
    if      ~isfield(mdSubj.formData, 'covid19') || ...
            ~exist(folderName + "\" + mdSubj.x_id + "\cough.wav", 'file') || ...
            ~isfield(mdSubj.formData.covid19, 'testDate') || ...
            ~isfield(mdSubj.formData.covid19, 'currentSymptoms')
        continue;
        
    % collect subject info
    else
        covid_status    = string(mdSubj.formData.covid19.diagnosedCovid19);
        PCR_test_date   = mdSubj.formData.covid19.testDate;
        test_symptoms   = mdSubj.formData.covid19.testSymptoms;
        recording_date  = mdSubj.datetime;
        age             = mdSubj.formData.age;
        gender          = mdSubj.formData.gender;
        current_symptoms = mdSubj.formData.covid19.currentSymptoms;
    end
    
    if covid_status == "Yes" % subject is positive
        posCount = posCount + 1;
        md_pos{posCount} = mdSubj;

        md_pos_table.subj_id(posCount)             = mdSubj.x_id;
        md_pos_table.PCR_test_date(posCount)       = PCR_test_date;
        md_pos_table.test_symptoms(posCount)       = strjoin(test_symptoms, ', ');
        md_pos_table.recording_date(posCount)      = recording_date(1:10);
        md_pos_table.current_symptoms(posCount)    = strjoin(current_symptoms, ', ');
        md_pos_table.age(posCount)                 = age;
        md_pos_table.gender(posCount)              = gender;
    end
end
