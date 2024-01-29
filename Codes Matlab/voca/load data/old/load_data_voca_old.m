%% load data

% load metadata + positive subjects metadata
% manually remove positive subjects who are not qualified
% load positive subjects cough audio files
% get statistics of positive subjects
% load negative subjects metadata
% load negative subjects cough audio files
% manually select negative subjects cough audio files
% get statistics of negative subjects
% plot all cough audio files

%% load metadata + positive subjects metadata

% remove previous data
close all; clc; clear;

% load metadata from database
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "\database\voca\voca-dataset-09-09-20";
md = jsondecode(fileread(folder_name + "\submissions.json")); % metadata
cd(prev_folder);


% pre-allocation
num_subj = length(md);
pos_count = 0;
md_pos = cell(num_subj,1); % metadata positive cell
num_col_table = 7;

md_pos_table = table( ...
    'Size', [num_subj num_col_table], ...
    ...
    'VariableTypes', {'string', 'string', 'string', 'string', 'string', ...
    'double', 'string'}, ...
    ...
    'VariableNames',{'subj_id', 'PCR_test_date', 'test_symptoms', 'recording_date', ...
    'current_symptoms', 'age', 'gender'} ...
    ); % metadata positive table

% md_pos_table:
% 1. subj id
% 2. PCR test date
% 3. PCR test symptoms
% 4. recording date
% 5. current symptoms
% 6. age
% 7. gender

% create metadata for positive subjects ==================================

% requirements:
% 1. covid status - exist
% 2. covid status == positive ("Yes")
% 3. cough file - exist
% 4. PCR test date - exist
% 5. current symptoms - exist

for i = 1 : num_subj % loop over all subjects in metadata
    
    md_subj = md{i}; % metadata of 1 subject
    
    % check requirements
    if      ~isfield(md_subj.formData, 'covid19') || ...
            ~exist(folder_name + "\" + md_subj.x_id + "\cough.wav", 'file') || ...
            ~isfield(md_subj.formData.covid19, 'testDate') || ...
            ~isfield(md_subj.formData.covid19, 'currentSymptoms')
        continue;
        
    % collect subject info
    else
        covid_status    = string(md_subj.formData.covid19.diagnosedCovid19);
        PCR_test_date   = md_subj.formData.covid19.testDate;
        test_symptoms   = md_subj.formData.covid19.testSymptoms;
        recording_date  = md_subj.datetime;
        age             = md_subj.formData.age;
        gender          = md_subj.formData.gender;
        current_symptoms = md_subj.formData.covid19.currentSymptoms;
    end
    
    if covid_status == "Yes" % subject is positive
        pos_count = pos_count + 1;
        md_pos{pos_count} = md_subj;

        md_pos_table.subj_id(pos_count)             = md_subj.x_id;
        md_pos_table.PCR_test_date(pos_count)       = PCR_test_date;
        md_pos_table.test_symptoms(pos_count)       = strjoin(test_symptoms, ', ');
        md_pos_table.recording_date(pos_count)      = recording_date(1:10);
        md_pos_table.current_symptoms(pos_count)    = strjoin(current_symptoms, ', ');
        md_pos_table.age(pos_count)                 = age;
        md_pos_table.gender(pos_count)              = gender;
    end
end

md_pos(pos_count + 1 : end) = [];
md_pos_table(pos_count + 1 : end, :) = [];

% save in file
save ../../ds/ds_load_voca md md_pos md_pos_table

% end of code run notification
disp('create metadata done!!');


%% manually remove positive subjects who are not qualified

% who is not qualified?
% 1. date - PCR test date and recording date are more then 2 weeks seperated.
% 2. symptoms - decision, based on test and current symptoms.

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% manualy chosen subjects to be removed

% num   id                          reason for removal

% 1     5e7720e211543b000819831e    date
% 4     5e7d7fe8233da300089e039e    symptoms
% 7     5e8202aec35c5200081234d2    date
% 11    5e8c678b7e8ff90007f3c608    symptoms
% 13    5e92edee0fbaff000745f501    bad audio signal    ***
% 16    5ea6acdc776b9200072eaf15    same subj as 15     ***
% 17    5eb2f7369e907b0007a97124    date + symptoms
% 19    5ee2023bf177dc0007c49bf2    symptoms
% 20    5ee756679c28020007b417c4    date + symptoms
% 21    5ee93d3921a5c80007ba5550    date + symptoms
% 24    5f158a3f975553000803b649    date + symptoms
% 25    5f189e21a3ee9b0008196321    symptoms
% 26    5f377ae970eb730008864f3c    date + symptoms
% 27    5f380959cab1e50007b16ac1    date + symptoms
% 28    5f380afacab1e50007b16ac2    date + symptoms
% 29    5f3d5f3958bf89000848c58b    symptoms
% 30    5f47d58c55a5960008ee2cdc    date + symptoms

% 1 4 7 11 13 16 17 19 20 21 24:30

remove_subj_indx = [1 ; 4 ; 7 ; 11 ; 13 ; 16 ; 17 ; 19 ; 20 ; 21 ; (24 : 30)'];

md_pos(remove_subj_indx) = [];
md_pos_table(remove_subj_indx, :) = [];

% save in file
save ../../ds/ds_load_voca md md_pos md_pos_table

% end of code run notification
disp('remove bad subjects done!!');

%% save positive subjects cough audio files

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% pre-allocation
subj_id = string(md_pos_table.subj_id);
num_subj_pos = length(subj_id);
num_col_table = 4;
num_class = 2;
fs = 16000;
audio_file_location = strings(num_subj_pos, 1);

ds = table( ...
    'Size', [num_subj_pos * num_class, num_col_table], ...
    'VariableTypes', {'cell', 'string', 'string', 'double'}, ...
    'VariableNames',{'audio_signal' 'subj_id' 'dataset' 'covid_status'} ...
    ); % voca data

% get folder name of all dataset
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "\database\voca\voca-dataset-09-09-20";
cd(prev_folder);

for i = 1 : num_subj_pos % loop over all positive subjects
    
    % set names of source and destination audio files
    source_file = folder_name + "\" + subj_id(i) + "\cough.wav";
    dest_file = ...
        folder_path + "DataBase\voca\" + "voca-audio-cough\" + i + ".wav"; % destination file
    
    % load cough audio file
    [s_t_raw, fs_raw] = audioread(source_file);
    
    % pre-process raw audio signal
    plot_flag = 0;
    sig_indx = i;
    s_t_pp = pre_process_audio_sig(s_t_raw, fs_raw, plot_flag, sig_indx);
    
    % save in table
    ds.audio_signal(i) = {s_t_pp};     % audio file
    ds.subj_id(i)      = subj_id(i);   % subject id
    ds.dataset(i)      = "voca";       % data source
    ds.covid_status(i) = 1;            % covid status
    
    % save audio to file
    s_t_pp = s_t_pp / max(abs(s_t_pp));
    audiowrite(dest_file, s_t_pp, fs);
    
    % save in another folder for audio data store
    dest_file = ...
        folder_path + "DataBase\voca\" + "audioDataFolder\positive\" + i + ".wav"; % destination file
    audiowrite(dest_file, s_t_pp, fs);
    audio_file_location(i) = dest_file;
end

% save metadata in excel file
metadata_voca_positive = table( ...
    audio_file_location, ...
    repmat("voca", num_subj_pos, 1), ...
    repmat("positive", num_subj_pos, 1), ...
    md_pos_table.age, ...
    md_pos_table.gender, ...
    'VariableNames',{'audio_file_location','ds','class','age','gender'});
writetable(metadata_voca_positive, '..\..\metadata\metadata_voca_positive.xlsx', 'WriteMode', 'replacefile');

% save in file
save ../../ds/ds_load_voca ds md md_pos md_pos_table

% end of code run notification
disp('save pos audio files done!!');


%% get statistics of positive subjects

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% pre-allocation
num_subj_pos = length(md_pos);
age = md_pos_table.age;
gender = md_pos_table.gender;
stat_pos = struct; % statistics positive

% get positive subjects statistical info
stat_pos.age_mean = mean(age); % age mean
stat_pos.age_std = std(age); % age std
stat_pos.age_range = [min(age) max(age)]; % min max age
stat_pos.num_male = sum(gender == "Male"); % No. male
stat_pos.num_female = sum(gender == "Female"); % No. female
stat_pos.geder_ratio = ...
    [stat_pos.num_male / num_subj_pos , ...
    stat_pos.num_female / num_subj_pos] * 100; % male/female ratio

% dispaly statistical info of negative subjects
disp(stat_pos);

% save in file
save ../../ds/ds_load_voca ds md md_pos md_pos_table stat_pos

% end of code run notification
disp('get statistics done!!');

%% load negative subjects metadata

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% pre-allocation
num_subj = length(md);
neg_count = 0;
md_neg = cell(num_subj,1); % metadata negative cell
num_col_table = 3;
age_min = stat_pos.age_range(1);
age_max = stat_pos.age_range(2);

md_neg_table = table( ...
    'Size', [num_subj num_col_table], ...
    'VariableTypes', {'string', 'double', 'string'}, ...
    'VariableNames',{'subj_id' 'age' 'gender'} ...
    ); % metadata negative table

% negative metadata table:
% 1. subj id
% 2. age
% 3. gender

% get folder name of all data
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "DataBase\voca\voca-dataset-09-09-20";
cd(prev_folder);


% create metadata for negative subjects ==================================

% requirements:
% 1. covid status - exist
% 2. covid status == negative ("No")
% 3. cough file - exist
% 4. currentSymptoms - exist + == 'I don’t have any symptom'
% 5. backgroundDiseases - exist + == 'I don’t have any of the above'
% 1. age range - same as for positive

for i = 1 : num_subj % loop over all subjects in metadata
    
    md_subj = md{i}; % metadata of 1 subject
    file_name = folder_name + "\" + md_subj.x_id + "\cough.wav"; % name of cough audio file
    
    % check requirements
    if      ~isfield(md_subj.formData, 'covid19') || ...
            ~exist(file_name, 'file') || ...
            (audioinfo(file_name).TotalSamples == 0) || ...
            ~isfield(md_subj.formData.covid19, 'currentSymptoms') || ...
            ~isfield(md_subj.formData, 'backgroundDiseases') || ...
            isempty(md_subj.formData.age) || ...
            ~isnumeric(md_subj.formData.age) || ...
            md_subj.formData.age < age_min || ...
            md_subj.formData.age > age_max
        continue;
    else
        covid_status        = string(md_subj.formData.covid19.diagnosedCovid19);
        current_symptoms    = md_subj.formData.covid19.currentSymptoms;
        background_diseases = md_subj.formData.backgroundDiseases;
        age                 = md_subj.formData.age;
        gender              = md_subj.formData.gender;
    end
    
    if      covid_status == "No" && ...
            min(strcmp(current_symptoms, {'I don’t have any symptom'})) && ...
            min(strcmp(background_diseases, {'I don’t have any of the above'}))
        
        neg_count = neg_count + 1;
        md_neg{neg_count} = md_subj;
        
        md_neg_table.subj_id(neg_count) = md_subj.x_id;
        md_neg_table.age(neg_count)     = age;
        md_neg_table.gender(neg_count)  = gender;
    end
end

md_neg(neg_count + 1 : end) = [];
md_neg_table(neg_count + 1 : end, :) = [];

% save in file
save ../../ds/ds_load_voca ds md md_pos md_pos_table stat_pos md_neg md_neg_table

% end of code run notification
disp('create negative metadata done!!');


%% copy negative subjects cough audio files to audio data store

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% pre-allocation
subj_id = md_neg_table.subj_id;
num_subj_neg = length(subj_id);
fs = 16000;
audio_file_location = strings(num_subj_neg, 1);

% get folder name of all data
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "DataBase\voca\voca-dataset-09-09-20";
cd(prev_folder);

for i = 1 : num_subj_neg % loop over all negative subjects
    
    % set names of source and destination audio files
    source_file = folder_name + "\" + subj_id(i) + "\cough.wav";
    
    % load cough audio file
    [s_t_raw, fs_raw] = audioread(source_file);
    
    % pre-process raw audio signal
    plot_flag = 0;
    sig_indx = i;
    s_t_pp = pre_process_audio_sig(s_t_raw, fs_raw, plot_flag, sig_indx);
    
    % save in another folder for audio data store
    s_t_pp = s_t_pp / max(abs(s_t_pp));
    dest_file = ...
        folder_path + "DataBase\voca\" + "audioDataFolder\negative\" + i + ".wav"; % destination file
    audiowrite(dest_file, s_t_pp, fs);
    audio_file_location(i) = dest_file;
end

% save metadata in excel file
metadata_voca_negative = table( ...
    audio_file_location, ...
    repmat("voca", num_subj_neg, 1), ...
    repmat("negative", num_subj_neg, 1), ...
    md_neg_table.age, ...
    md_neg_table.gender, ...
    'VariableNames',{'audio_file_location','ds','class','age','gender'});
writetable(metadata_voca_negative, '..\..\metadata\metadata_voca_negative.xlsx', 'WriteMode', 'replacefile');

% end of code run notification
disp('save neg audio files done!!');

%% manually select negative subjects cough audio files

% requirement:
% cough audio recording looks OK
% male\female ratio - same as for positive subjects

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% pre-allocation
subj_id = md_neg_table.subj_id;
num_subj_neg = length(subj_id);
keep_subj_indx = [3 ; 5 ; 7 ; 14 ; 26 ; 30 ; 31 ; 44 ; 48 ; 61 ; 71 ; 86 ; 105];
fs = 16000;

% get folder name of all data
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "DataBase\voca\voca-dataset-09-09-20";
cd(prev_folder);


for i = 1 : num_subj_neg % loop over all negative subjects
    
    % set names of source and destination audio files
    source_file = folder_name + "\" + subj_id(i) + "\cough.wav";
    dest_file = ...
        folder_path + "DataBase\voca\" + "voca-audio-cough-neg\" + i + ".wav";
    
    % load cough audio file
    [s_t_raw, fs_raw] = audioread(source_file);
    
    % pre-process raw audio signal
    plot_flag = 0;
    sig_indx = i;
    s_t_pp = pre_process_audio_sig(s_t_raw, fs_raw, plot_flag, sig_indx);
    
    % save audio to file
    s_t_pp = s_t_pp / max(abs(s_t_pp));
    audiowrite(dest_file, s_t_pp, fs);
end

% male: 3 5 7 14 30
%%% female: 26 31 44 48 61 63 71 86
% female: 26 31 44 48 61 71 86 105
% total: 3 5 7 14 26 30 31 44 48 61 71 86 105

% keep chosen subjects in metadata
md_neg = md_neg(keep_subj_indx);
md_neg_table = md_neg_table(keep_subj_indx, :);

% save in file
save ../../ds/ds_load_voca ds md md_pos md_pos_table stat_pos md_neg md_neg_table

% end of code run notification
disp('keep good subjects done!!');


%% load negative subjects cough audio files

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% pre-allocation
subj_id = md_neg_table.subj_id;
num_subj_neg = length(subj_id);
fs = 16000;

% get folder name of all data
prev_folder = cd('..\..\..\..\');
folder_path = pwd;
folder_name = folder_path + "DataBase\voca\voca-dataset-09-09-20";
cd(prev_folder);

for i = 1 : num_subj_neg % loop over all negative subjects
    
    % set names of source and destination audio files
    source_file = folder_name + "\" + subj_id(i) + "\cough.wav";
    dest_file = ...
        folder_path + "DataBase\voca\" + "voca-audio-cough\" + (num_subj_neg+i) + ".wav";
    
    % load cough audio file
    [s_t_raw, fs_raw] = audioread(source_file);
    
    % pre-process raw audio signal
    plot_flag = 0;
    sig_indx = i;
    s_t_pp = pre_process_audio_sig(s_t_raw, fs_raw, plot_flag, sig_indx);
    
    % save in table
    ds.audio_signal(i + num_subj_neg) = {s_t_pp};       % audio file
    ds.subj_id(i + num_subj_neg)      = subj_id(i);     % subject id
    ds.dataset(i + num_subj_neg)      = "voca";         % data source
    ds.covid_status(i + num_subj_neg) = 0;              % covid status
    
    % save audio to file
    s_t_pp = s_t_pp / max(abs(s_t_pp));
    audiowrite(dest_file, s_t_pp, fs);
end

% save in file
save ../../ds/ds_load_voca ds md md_pos md_pos_table stat_pos md_neg md_neg_table

% end of code run notification
disp('load neg audio files done!!');


%% get statistics of negative subjects

% remove previous data
close all; clc; clear;

% load data
load ../../ds/ds_load_voca

% pre-allocation
num_subj_neg = length(md_neg);
age = md_neg_table.age;
gender = md_neg_table.gender;
stat_neg = struct; % statistics negative

% get negative subjects statistical info
stat_neg.age_mean = mean(age); % age mean
stat_neg.age_std = std(age); % age std
stat_neg.age_range = [min(age) max(age)]; % min max age
stat_neg.num_male = sum(gender == "Male"); % No. male
stat_neg.num_female = sum(gender == "Female"); % No. female
stat_neg.geder_ratio = ...
    [stat_neg.num_male / num_subj_neg , ...
    stat_neg.num_female / num_subj_neg] * 100; % male/female ratio

% dispaly statistical info of negative subjects
disp(stat_neg);

% save in file
save ../../ds/ds_load_voca ds md md_pos md_pos_table stat_pos md_neg md_neg_table stat_neg

% end of code run notification
disp('get statistical data done!!');


%% plot all cough audio files

% % remove previous data
% close all; clc; clear;
% 
% % load data
% load ../../ds/ds_load_voca
% 
% for i = 1 : 26
%     
%     s_t = ds.audio_signal{i};
%     fs = 16000;
%     time = (0 : length(s_t) - 1) / fs;
%     
%     figure;
%     plot(time, s_t);
% end

