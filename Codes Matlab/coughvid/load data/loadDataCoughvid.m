%% load data

% process:
% 1. load table of metaData.
% 2. remove irelevent parts from metaData.
% 3. remove bad subjects.
% 4. load positive and negative subjects to folder + update metaData

%% load metaData

% remove previous data
close all; clc; clear;

% load metadata from database
prevFolder = cd('..\..\..\..\');
folderPath = pwd;
folderName = folderPath + "DataBase\coughvid\coughvid-dataset-03-02-21";
metaDataOrigin = readtable(folderName + "\metadata_compiled.csv"); % metadata
cd(prevFolder);

% save in file
save coughvidInfo metaDataOrigin

% end of code run notification
disp('load metadata done!!');

%% remova bad columns from metaData

% remove previous data
close all; clc; clear;

% load data
load coughvidInfo

% columns to keep:
% 1. id
% 3. cough detected (chance to have cough in signal)
% 7. age
% 8. gender
% 11. covid status
% 12 22 32 42. cough quality
% 20 30 40 50. diagnosis

metaDataOrigin = metaDataOrigin(:, [1 3 7 8 11 12 20 22 30 32 40 42 50]);

% save in file
save coughvidInfo metaDataOrigin

% end of code run notification
disp('remova extra info done!!');

%% remove bad subjects metaData

% remove previous data
close all; clc; clear;

% load data
load coughvidInfo

% see all possible options for each category
optStatus = unique(metaDataOrigin.status); % covid status
optQuality = unique(metaDataOrigin.quality_1); % cough quality
optDiagnosis = unique(metaDataOrigin.diagnosis_1); % covid diagnosis

% save options as strings
status = string(metaDataOrigin.status);
age = metaDataOrigin.age;
quality_1 = string(metaDataOrigin.quality_1);
quality_2 = string(metaDataOrigin.quality_2);
quality_3 = string(metaDataOrigin.quality_3);
quality_4 = string(metaDataOrigin.quality_4);
diagnosis_1 = string(metaDataOrigin.diagnosis_1);
diagnosis_2 = string(metaDataOrigin.diagnosis_2);
diagnosis_3 = string(metaDataOrigin.diagnosis_3);
diagnosis_4 = string(metaDataOrigin.diagnosis_4);

% find positive indices based on strings
indxStatusPos = status == "COVID-19";
indxStatusNeg = status == "healthy";
indxAgeBad = isnan(age) | age < 10 | age > 90;
indxQualityGood = ...
    quality_1 == "good" | quality_1 == "ok" | ...
    quality_2 == "good" | quality_2 == "ok" | ...
    quality_3 == "good" | quality_3 == "ok" | ...
    quality_4 == "good" | quality_4 == "ok";
indxQualityBad = ...
    quality_1 == "no_cough" | quality_1 == "poor" | ...
    quality_2 == "no_cough" | quality_2 == "poor" | ...
    quality_3 == "no_cough" | quality_3 == "poor" | ...
    quality_4 == "no_cough" | quality_4 == "poor";
indxDiagnosisPos = ...
    diagnosis_1 == "COVID-19" | ...
    diagnosis_2 == "COVID-19" | ...
    diagnosis_3 == "COVID-19" | ...
    diagnosis_4 == "COVID-19";
% indxDiagnosisNeg = ...
%     diagnosis_1 == "healthy_cough" | diagnosis_1 == "lower_infection" | ...
%     diagnosis_1 == "obstructive_disease" | diagnosis_1 == "upper_infection" | ...
%     diagnosis_2 == "healthy_cough" | diagnosis_2 == "lower_infection" | ...
%     diagnosis_2 == "obstructive_disease" | diagnosis_2 == "upper_infection" | ...
%     diagnosis_3 == "healthy_cough" | diagnosis_3 == "lower_infection" | ...
%     diagnosis_3 == "obstructive_disease" | diagnosis_3 == "upper_infection" | ...
%     diagnosis_4 == "healthy_cough" | diagnosis_4 == "lower_infection" | ...
%     diagnosis_4 == "obstructive_disease" | diagnosis_4 == "upper_infection";
indxDiagnosisNeg = ...
    diagnosis_1 == "healthy_cough" | ...
    diagnosis_2 == "healthy_cough" | ...
    diagnosis_3 == "healthy_cough" | ...
    diagnosis_4 == "healthy_cough";
indxDiagnosisBad = ...
    diagnosis_1 == "lower_infection" | diagnosis_1 == "obstructive_disease" | diagnosis_1 == "upper_infection" | ...
    diagnosis_2 == "lower_infection" | diagnosis_2 == "obstructive_disease" | diagnosis_2 == "upper_infection" | ...
    diagnosis_3 == "lower_infection" | diagnosis_3 == "obstructive_disease" | diagnosis_3 == "upper_infection" | ...
    diagnosis_4 == "lower_infection" | diagnosis_4 == "obstructive_disease" | diagnosis_4 == "upper_infection";

% combine all options
indxPos = ...
    (indxStatusPos .* (1-indxAgeBad) .* indxQualityGood .* (1-indxQualityBad) .* ...
    indxDiagnosisPos .* (1-indxDiagnosisNeg) .* (1-indxDiagnosisBad)) > 0;
indxNeg = ...
    (indxStatusNeg .* (1-indxAgeBad) .* indxQualityGood .* (1-indxQualityBad) .* ...
    indxDiagnosisNeg .* (1-indxDiagnosisPos) .* (1-indxDiagnosisBad)) > 0;

% update metaData
metaDataPos = metaDataOrigin(indxPos, :);
metaDataPos.class = repmat("positive", height(metaDataPos), 1);
metaDataNeg = metaDataOrigin(indxNeg, :);
metaDataNeg.class = repmat("negative", height(metaDataNeg), 1);
metaData = [metaDataNeg ; metaDataPos];

% save in file
save coughvidInfo metaDataOrigin metaData metaDataPos metaDataNeg

% end of code run notification
disp('remove bad subjects done!!');

%% load subjects cough audio files

% remove previous data
close all; clc; clear;

% load data
load coughvidInfo

% pre-allocation
subjId = string(metaData.uuid);
numSubj = length(subjId);
numColTable = 4;
numClasses = 2;
fs = 16000;
audioFileLocation = strings(numSubj, 1);

% create table of dataset
ds = table( ...
    'Size', [numSubj * numClasses, numColTable], ...
    'VariableTypes', {'cell', 'string', 'string', 'double'}, ...
    'VariableNames',{'audio_signal' 'subj_id' 'dataset' 'covid_status'} ...
    ); % coughvid data

% get folder name of all data
prevFolder = cd('..\..\..\..\');
folderPath = pwd;
folderName = folderPath + "DataBase\coughvid\";
cd(prevFolder);

% loop over all subjects
for row = 1 : numSubj
    
    % copy cough audio file from source to destination
    sourceFile = [...
        folderName + "coughvid-dataset-03-02-21-WAV+OGG" + "\" + subjId(row) + ".wav" ; ...
        folderName + "coughvid-dataset-03-02-21-WAV+OGG" + "\" + subjId(row) + ".ogg"
        ];
    
    indxToKeep = [exist(sourceFile(1), 'file') exist(sourceFile(2), 'file')] > 0;
    sourceFile = sourceFile(indxToKeep);
    
    % load cough audio file
    [s_t_raw, fs_raw] = audioread(sourceFile);
    s_t_raw = s_t_raw(:,1);
    
    % pre-process raw audio signal
    plot_flag = 0;
    sig_indx = row;
    s_t_pp = pre_process_audio_sig(s_t_raw, fs_raw, plot_flag, sig_indx);
    
    % save audio to file
    s_t_pp = s_t_pp / max(abs(s_t_pp));
    class = metaData.class(row);
    dest_file = folderName + "audioDataFolderVer1\" + class + "\" + subjId(row) + ".wav";
    audiowrite(dest_file, s_t_pp, fs);
    audioFileLocation(row) = dest_file;
end

% add info to metaData
metaData.audioFileLocation = audioFileLocation;
metaData.dataset = repmat("coughvid", height(metaData), 1);

% save metaData in csv file
writetable(metaData, '..\..\metadata\metaDataCoughvid.xlsx', 'WriteMode', 'replacefile');

% save in file
save coughvidInfo metaDataOrigin metaData metaDataPos metaDataNeg

% end of code run notification
disp('load audio files done!!');

%% get statistics of subjects

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load coughvidInfo

dsStatistics = extract_statistics(metaData);

% save in folder
save coughvidInfo metaDataOrigin metaData metaDataPos metaDataNeg dsStatistics

% end of code run notification
disp('get statistics done!!');
