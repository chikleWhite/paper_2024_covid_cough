%% remove bad subjects

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

PosSymptomsYes = ["positive_mild" "positive_moderate" "resp_illness_not_identified"];
PosSymptomsNo = "positive_asymp";
NegSymptomsYes = "resp_illness_not_identified";
NegSymptomsNo = ["healthy" "no_resp_illness_exposed"];

posIndx = metadata.class == "positive";
negIndx = metadata.class == "negative";

metadataPosSymptomsYes = metadata(ismember(string(metadata.covid_status), PosSymptomsYes) & posIndx, :);
metadataPosSymptomsNo = metadata(ismember(string(metadata.covid_status), PosSymptomsNo) & posIndx, :);
metadataNegSymptomsYes = metadata(ismember(string(metadata.covid_status), NegSymptomsYes) & negIndx, :);
metadataNegSymptomsNo = metadata(ismember(string(metadata.covid_status), NegSymptomsNo) & negIndx, :);

% display No. subjects
disp(["No. subjects (PosSymptomsYes): " height(metadataPosSymptomsYes)]);
disp(["No. subjects (PosSymptomsNo): " height(metadataPosSymptomsNo)]);
disp(["No. subjects (NegSymptomsYes): " height(metadataNegSymptomsYes)]);
disp(["No. subjects (NegSymptomsNo): " height(metadataNegSymptomsNo)]);