%% load data

% process:
% 1. load table of metadata.
% 2. remove irelevent parts from metadata.
% 3. find positive subjects.
% 4. copy positive audio files to different folder.
% (those files are manually converted to wav format because matlab can't do it..).
% 5. create table of all positive subjects.
% 6. get positive subjects statistics.
% 7. do the same for negative subjects.

%% load metadata

% remove previous data
close all; clc; clear;

% load metadata from database
prev_folder = cd('..\..\..\..\');
folder_path = pwd + "\DataBase\coughvid";
folder_name = folder_path + "\coughvid-dataset-03-02-21-WAV+OGG";
metadata = readtable(folder_path + "\metadata_compiled.csv"); % metadata
cd(prev_folder);

% take only 1t row to observe
md_1st_row = metadata(1,:);

% keep only subjects with cough_detected >= 0.7
metadata(metadata.cough_detected < 0.7, :) = [];

% see all possible options for status
status_opt = unique(metadata.status); % covid status

% get status info
status_raw = string(metadata.status);
num_0 = sum(status_raw == "");
num_symptomatic = sum(status_raw == "symptomatic");
num_COVID = sum(status_raw == "COVID-19");
num_healthy = sum(status_raw == "healthy");

% keep only subjects with status "healthy" or "COVID-19"
status_bad = status_raw ~= 'COVID-19' & status_raw ~= 'healthy';
metadata(status_bad, :) = [];

% extract statistical information from metadata
exstract_metadata_info(metadata);

% pre-allocation prior pre-processing and coping all audio files
num_subj = size(metadata, 1);
subj_id = string(metadata.uuid);
status = string(metadata.status);
pos_count = 0;
neg_count = 0;
fs = 16000;

tic;

% loop over all subjects
for i = 1 : num_subj
    
    % check status -> update destination location and file number
    switch status(i)
        
        case "COVID-19" % positive
            
            pos_count = pos_count + 1;
            dest_loc = "positive";
            
        case "healthy" % negative
            
            neg_count = neg_count + 1;
            dest_loc = "negative";
    end
    
    % name of source file (wav or ogg)
    source_file_wav = folder_name + "\" + subj_id(i) + ".wav"; % wav format
    source_file_ogg = folder_name + "\" + subj_id(i) + ".ogg"; % ogg format
    
    % name of destination file (wav)
    dest_file = folder_path + "\" + "audioDataFolderNoRaters" + "\" + dest_loc + "\" + subj_id(i) + ".wav";
    
    % load audio file (check both options - wav/ogg)
    if exist(source_file_wav, 'file') % wav format
        
        [s_t_raw, fs_raw] = audioread(source_file_wav);
        
    elseif exist(source_file_ogg, 'file') % ogg format
        
        [s_t_raw, fs_raw] = audioread(source_file_ogg);
    end
    
    % if there are 2 channels (srereo) -> keep only 1 (mono)
    s_t_raw = s_t_raw(:,1);
    
    % pre-process raw audio signal
    plot_flag = 0;
    sig_indx = i;
    s_t_pp = pre_process_audio_sig(s_t_raw, fs_raw, plot_flag, sig_indx);
    
    % save audio to file
    s_t_pp = s_t_pp / max(abs(s_t_pp));
    audiowrite(dest_file, s_t_pp, fs);
    
    % remove previous text from command window
    clc;
    
    % show current iter number and percent done
    disp(['work done: ', num2str(i), ' iters out of ', num2str(num_subj), ' (', num2str(round(i / num_subj * 100, 2)), ' %)']);
    
    % show time elapsed
    hours = floor(toc/3600);
    minuts = floor(toc/60) - hours * 60;
    seconds = floor(toc - hours * 3600  - minuts * 60);
    disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);
    
    % show time remain
    time_remain = toc / i * (num_subj - i);
    hours = floor(time_remain/3600);
    minuts = floor(time_remain/60) - hours * 60;
    seconds = floor(time_remain - hours * 3600  - minuts * 60);
    disp(['time remain ~ ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);
end
