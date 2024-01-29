%% check correlation between pcr results and raters annotations

%% load relevent data files

% files:
% metadata - information about all subjects

% remove previous data
close all; clc; clear;

% pre-allocation
specs = struct;

% load metadata from database
prev_folder = cd('..\..\..\');
folder_path = pwd;
folder_name = folder_path + "DataBase\coughvid\coughvid-dataset-03-02-21";
metaDataOrigin = readtable(folder_name + "\metadata_compiled.csv"); % metadata
cd(prev_folder);

% create new metadata with only relevent information
metaData = ...
    metaDataOrigin(:, {'status', 'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4'});

% keep only subjects with experts annotations (at least 1)
expertsAnnotaions = ...
    [metaData.diagnosis_1 metaData.diagnosis_2 metaData.diagnosis_3 metaData.diagnosis_4];
AnnotationIndices = sum(~ismissing(expertsAnnotaions), 2) > 0;
metaData = metaData(AnnotationIndices, :);

% remove subjects without status
metaData(ismissing(metaData.status), :) = [];

% see all possible options for each category
opt_status = unique(metaData.status); % covid status
opt_diagnosis = unique(metaData.diagnosis_1); % covid diagnosis

% remove following status:
% subjects: "symptomatic".
% raters: "upper infection" | "obstructive disease" | "lower infection".
badDiagnosis = ["upper_infection" "obstructive_disease" "lower_infection"];
metaData(metaData.status == "symptomatic", :) = [];
metaData(ismember(string(metaData.diagnosis_1), badDiagnosis), :) = [];
metaData(ismember(string(metaData.diagnosis_2), badDiagnosis), :) = [];
metaData(ismember(string(metaData.diagnosis_3), badDiagnosis), :) = [];
metaData(ismember(string(metaData.diagnosis_4), badDiagnosis), :) = [];

% correlation check between status and diagnosis:

% create correlation matrix
correlationMat = zeros(size(metaData));

% change status/diagnosis from string to numerical values ([0, 1, 2]):

% 0 - healthy
% 1 - status: Symptomatic but no COVID-19 diagnosis
% 1 - diagnosis: Lower infection / obstructive disease / upper infection
% 2 - COVID-19

% loop over all columns correlationMat
for col = 1 : width(correlationMat)
    correlationMat(:, col) = string2categorical(string(metaData{:, col}));
end

% specs struct
specs.numSubjectsOrigin = height(metaDataOrigin);
specs.numSubjects = height(metaData);
specs.numDiagnoses = sum(~isnan(correlationMat(:, 2 : end)), 'all');
specs.numDiagnosesNeg = sum(correlationMat(:, 2 : end) == 0, 'all');
specs.numDiagnosesPos = sum(correlationMat(:, 2 : end) == 1, 'all');

% compare between status and diagnosis all
specs.numDiagnosesCorrect = ...
    sum(correlationMat(:, 2 : end) == correlationMat(:, 1), 'all');
specs.prcntDiagnosesCorrect = round(specs.numDiagnosesCorrect / specs.numDiagnoses * 100, 1);

specs.numDiagnosesNegCorrect = sum(correlationMat(:, 2 : end) == 0 & ...
    correlationMat(:, 2 : end) == correlationMat(:, 1), 'all');
specs.prcntDiagnosesNegCorrect = round(specs.numDiagnosesNegCorrect / specs.numDiagnosesNeg * 100, 1);

specs.numDiagnosesPosCorrect = sum(correlationMat(:, 2 : end) == 1 & ...
    correlationMat(:, 2 : end) == correlationMat(:, 1), 'all');
specs.prcntDiagnosesPosCorrect = round(specs.numDiagnosesPosCorrect / specs.numDiagnosesPos * 100, 1);

% display final specs
disp(specs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% change vector from string to numerical value ([0, 1, 2])
function categoricalVec = string2categorical(stringVec)
    
    % pre-allocation
    categoricalVec = stringVec;
    
    % healthy => 0
    categoricalVec(...
        stringVec == "healthy" | ...
        stringVec == "healthy_cough") = 0;
    
    % COVID-19 => 1
    categoricalVec(...
        stringVec == "COVID-19") = 1;
end

% % change vector from string to numerical value ([0, 1, 2])
% function categoricalVec = string2categorical(stringVec)
%     
%     % pre-allocation
%     categoricalVec = stringVec;
%     
%     % healthy => 0
%     categoricalVec(...
%         stringVec == "healthy" | ...
%         stringVec == "healthy_cough") = 0;
%     
%     % symptomatic => 1
%     categoricalVec(...
%         stringVec == "symptomatic" | ...
%         stringVec == "lower_infection" | ...
%         stringVec == "obstructive_disease" | ...
%         stringVec == "upper_infection") = 1;
%     
%     % COVID-19 => 2
%     categoricalVec(...
%         stringVec == "COVID-19") = 2;
% end
