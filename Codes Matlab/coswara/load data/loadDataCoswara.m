%% Load coswara dataset

% overview

%% load relevent data files

% files:
% metadata - information about all subjects
% codebook - explenation to what each column mean
% annotations - quality scores to audio files

% remove previous data
close all; clc; clear;

% set folder path
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "\DataBase\coswara\Coswara-dataset-9.8.22-tar.gz\";

% set file names
file_name_codebook = "csv_labels_legend.json";
file_name_metadata = "combined_data.csv";
file_name_audio_quality_breath_deep = "breathing-deep_labels_pravinm.csv";
file_name_audio_quality_breath_shallow = "breathing-shallow_labels_pravinm.csv";
file_name_audio_quality_cough_heavy = "cough-heavy_labels_debottam.csv";
file_name_audio_quality_cough_shallow = "cough-shallow_labels_debarpan.csv";
cd(prev_folder);

% load files to matlab
metadata_origin = readtable(folder_name + file_name_metadata); % metadata
metadata_codebook = jsondecode(fileread(folder_name + file_name_codebook)); % code book for metadata

% audio quality annotations
audio_quality_breath_deep = readtable(folder_name + "annotations\" + file_name_audio_quality_breath_deep); % audio_quality_breath_deep
audio_quality_breath_shallow = readtable(folder_name + "annotations\" + file_name_audio_quality_breath_shallow); % audio_quality_breath_shallow
audio_quality_cough_heavy = readtable(folder_name + "annotations\" + file_name_audio_quality_cough_heavy); % audio_quality_cough_heavy
audio_quality_cough_shallow = readtable(folder_name + "annotations\" + file_name_audio_quality_cough_shallow); % audio_quality_cough_shallow

% put all annotations in 1 table
audio_quality = table({audio_quality_breath_deep}, {audio_quality_breath_shallow}, ...
    {audio_quality_cough_heavy}, {audio_quality_cough_shallow}, ...
    'VariableNames', {'audio_quality_breath_deep', 'audio_quality_breath_shallow', ...
    'audio_quality_cough_heavy', 'audio_quality_cough_shallow'});

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality

% end of code run notification
disp('load metadata done!!');

%% combine metadata and audio quality annotations and keep only relevent columns in metadata

% relevent info:

% 1. id                - user id
% 2. a                 - age
% g                 - Gender (male/female/other)
% record_date       - Date when the user recorded and submitted the samples
% covid_status      - Health status (e.g. : positive_mild, healthy,etc.)

% testType          - Type of test (RAT/RT-PCR)
% test_date         - Date of COVID Test (if taken)
% test_status       - Status of COVID Test (p->Positive, n->Negative, na-> Not taken Test)

% ctScan            - CT-Scan (y/n if the user has taken a test)
% ctDate            - Date of CT-Scan
% ctScore           - CT-Score

% Chest CT severity score calculation:
% based on degree of involvement of the lung lobes as 0%, (0 points),
% 1-25% (1 point), 26-50% (2 points), 51-75% (3 points), and 76-100% (4 points).
% The CT severity score was quantified by summing the 5 lobe indices.
% Overall result range is 0-20 (4*5).

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

% create new metadata with only relevent information
metadata = metadata_origin(:, {'id', 'a', 'g', 'record_date', 'covid_status', ...
    'testType', 'test_date', 'test_status', 'ctScan', 'ctDate', 'ctScore'});

% add new columns: folder original location, folder new location (after pre-processing), dataset, class
metadata.folder_origin = repmat("", height(metadata), 1); % folder origin - pre-allocation
metadata.folder = repmat("", height(metadata), 1); % folder - pre-allocation
metadata.dataset = repmat("coswara", height(metadata), 1); % dataset

% add class based on pcr test results
metadata.class = metadata.test_status; % class
metadata.class(string(metadata.class) == "n") = {'negative'};
metadata.class(string(metadata.class) == "p") = {'positive'};

% change order: [subject id, folder location, dataset, class, everything else]
metadata = [metadata(:,1) metadata(:, end-3) metadata(:, end-2) metadata(:, end-1) metadata(:, end) metadata(:,2:end-4)];

% add new columns for audio quality annotations
% categories: 2 (excellent), 1 (good), 0 (bad), -1 (not labeled)
% 1st: set all to not labeled
metadata.audio_quality_breath_deep =    ones(height(metadata), 1) * -1;
metadata.audio_quality_breath_shallow = ones(height(metadata), 1) * -1;
metadata.audio_quality_cough_heavy =    ones(height(metadata), 1) * -1;
metadata.audio_quality_cough_shallow =  ones(height(metadata), 1) * -1;

% names of audio quality columns
col_names = ["audio_quality_breath_deep" "audio_quality_breath_shallow" ...
    "audio_quality_cough_heavy" "audio_quality_cough_shallow"];

% loop over all 4 audio types (breath deep/shallow, cough heavy/shallow)
for col_name = col_names
    
    % per col, recieve subjects ids and audio qualities
    audio_name = erase(col_name, "audio_quality");
    row_names = string(audio_quality.(col_name){1}.FILENAME);
    row_names = extractBefore(row_names,29);
    row_qualities = audio_quality.(col_name){1}.QUALITY;
    
    % loop over all subjects per audio type
    for subj_row = 1 : length(row_names)
        id_indx = find(contains(string(metadata.id), row_names(subj_row)));
        metadata.(col_name)(id_indx) = row_qualities(subj_row); % update metadata
    end
end

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality metadata

% end of code run notification
disp('combine metadata and audio quality annotations and keep relevent info done!!');

%% remove bad subjects

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

% remove subjects without test_status information
% search for 'ut' (na-> Not taken Test) and remove
metadata((metadata.test_status ~= "p" & metadata.test_status ~= "n"), :) = [];

% remove subjects without testType information or did test other then PCR
metadata((metadata.testType == "" | metadata.testType ~= "rtpcr"), :) = [];

% remove subjects with time interval between record date and test date
% higher then 2 weeks
recordDateNum = datenum(metadata.record_date);
pcrTestDateNum = datenum(metadata.test_date);
timeIntervalThreshold = 14; % days
badTimeInterval = (recordDateNum - pcrTestDateNum) > timeIntervalThreshold;
metadata(badTimeInterval, :) = [];

% remove subjects without audio quality annotation or with bad quality results
% metadata((any(table2array(metadata(:, end-3 : end)) == -1, 2) | ...
%     any(table2array(metadata(:, end-3 : end)) == 0, 2)), :) = [];

% remove subjects with bad quality results (= 0)
metadata((any(table2array(metadata(:, end-3 : end)) == 0, 2)), :) = [];

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality metadata

% end of code run notification
disp('remove bad subjects done!!');

%% find subfolder of each subject and add to metadata

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

% get folder locations of Coswara-dataset-9.8.22
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "\DataBase\coswara\Coswara-dataset-9.8.22\";
cd(prev_folder);

% get subfolders location
sub_folders_names = string(ls(folder_name));
sub_folders_names(1:2) = [];
sub_folders_names = folder_name + sub_folders_names;

% names of all subjects
subj_names = string(metadata.id);

% loop over all subjects
for subj_row = 1 : length(subj_names)
    
    % loop over all subfolders to find subfolder of each subject
    for sub_folder_row = 1 : length(sub_folders_names)
        
        % names of all files in subfolder
        sub_folder_name = sub_folders_names(sub_folder_row);
        files_names = string(ls(sub_folder_name));
        
        % check if subfolder contains subject -> add folder location to
        % metadata
        if any(strcmp(files_names, subj_names(subj_row)))
            metadata.folder_origin(subj_row) = sub_folder_name;
        end
    end
end

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality metadata

% end of code run notification
disp('add subfolders locations to metadata done!!');

%% add another col to metadata (symptoms: yes/no)

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

% add new column
metadata.symptoms = repmat("", height(metadata), 1); % symptoms (yes/no)

% names of symptoms
symptomsYes = ["positive_mild" "positive_moderate" "resp_illness_not_identified"];
symptomsNo = ["positive_asymp" "no_resp_illness_exposed" "healthy"];

% update symptoms yes/no col
metadata.symptoms(ismember(string(metadata.covid_status), symptomsYes)) = "yes";
metadata.symptoms(ismember(string(metadata.covid_status), symptomsNo)) = "no";

% change order: [subject id, folder location, dataset, class, everything else]
metadata = [metadata(:, 1 : 5) metadata(:, end) metadata(:, 6 : end - 1)];

% save final metadata in CSV file
writetable(metadata, 'metadata coswara post subjFilt-NI.xlsx');

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality metadata

% end of code run notification
disp('add symptoms col to metadata done!!');

%% Nadav and Nir annotations - update metadata and remove bad subjects

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

% load metadata csv file with Nadav's annotations
file_name = "metadata coswara post subjFilt-NI annotate-NA annotate-NI";
metadata = readtable(file_name);

% Nadav's and Nir's annotations: "Good" | "NoGood" | "" (if not annotated)
OurAnnotations = string(metadata{:, end - 3 : end});

% remove bad subjects
% non of 4 audio types is "Good"
numBadAudiosPerSubj = sum(OurAnnotations == "NoGood" | OurAnnotations == "", 2);
metadata(numBadAudiosPerSubj == 4, :) = [];
OurAnnotations(numBadAudiosPerSubj == 4, :) = [];

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality metadata OurAnnotations

% end of code run notification
disp('remove bad subjects based on nadav annotations done!!');

%% pre-processing and copy audio files to new folder sorted by negative/positive

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

% pre-allocation
fs = 16000;
metadata.folder = strings(height(metadata), 1);

% names of audio types
audio_names = ["breathing-deep" "breathing-shallow" "cough-heavy" "cough-shallow"];

% loop over all subjects
for row = 1 : height(metadata)
    
    % loop over all audio types per subject
    for audio_num = 1 : 4
        
        % check audio file quality based on Nadav's annotations
        switch OurAnnotations(row, audio_num)
            
            case "NoGood"
                
                continue; % skip audio file without copy to new location
                
            case "Good"
                
                % continue with code (copy audio file to new location)
        end
        
        % audio file source location
        source_file = metadata.folder_origin(row) + "\" + metadata.id(row) + ...
            "\" + audio_names(audio_num) + ".wav";
        
        % load cough audio file
        [audio_signal_raw, fs_raw] = audioread(source_file);
        
        % pre-process raw audio signal
        plot_flag = 0; % plot?
        audio_signal_pp = pre_process_audio_sig(audio_signal_raw, fs_raw, plot_flag, row, audio_num);
        
        % save audio to file in destination
        
        % change folder location to destination
        str_len = length(char(metadata.folder_origin(row)));
        new_str = extractBefore(metadata.folder_origin(row),str_len - 30);
        
        % choose folder location based on class
        switch metadata.class{row}
            
            case "positive"
                
                metadata.folder(row) = new_str + "audioDataFolder\positive\";
                
            case "negative"
                
                metadata.folder(row) = new_str + "audioDataFolder\negative\";
        end
        
        % folder name of destination file
        destination_file = metadata.folder(row) + metadata.id{row} + ...
            "\" + audio_names(audio_num) + ".wav";
        
                % check if folder of subject id doesnt exist -> create new id folder
                if ~isfolder(metadata.folder(row) + metadata.id{row})
                    mkdir(metadata.folder(row) + metadata.id{row})
                end
        
                % write audio file to destination
                audiowrite(destination_file, audio_signal_pp, fs);
    end
end

% save final metadata in CSV file
writetable(metadata, 'metadata coswara final.csv');

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality metadata OurAnnotations

% end of code run notification
disp('add subfolders locations to metadata done!!');

%% get statistics of subjects

% remove previous data
close all; clc; clear;

% load metadata files from previous sections
load meatadata_files_coswara

ds_statistics = extract_statistics(metadata);

% save in folder
save meatadata_files_coswara metadata_origin metadata_codebook audio_quality metadata OurAnnotations ds_statistics

% end of code run notification
disp('get statistics done!!');
