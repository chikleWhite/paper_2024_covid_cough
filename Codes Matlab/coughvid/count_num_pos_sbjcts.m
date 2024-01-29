%% load metadata

% remove previous data
close all; clc; clear;

% load metadata from database
prev_folder = cd('..\..\..\');
folder_path = pwd;
folder_name = folder_path + "\database\coughvid\coughvid-dataset-03-02-21";
md = readtable(folder_name + "\metadata_compiled.csv"); % metadata
cd(prev_folder);

status = string(md.status);
indx_status_pos = status == "COVID-19";
num_pos_sbj  = sum(indx_status_pos);