function [Features, trueLabels, adsSpecs, win_info] = load_ds()

tic;

% Create audioDatastore object of the data

% load coughvidNoRaters dataset
ds_name = "coughvid";
folder_name = "../../../DataBase/" + ds_name + "/" + "audioDataFolderNoRaters";
ads_coughvidNoRaters = audioDatastore(folder_name, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

% pre-processing

% parameters for audio feature extractor (mel-spectrogram)
% win & spectrogram length/overlap info
win_info.win_len = 128;
win_info.overlap_coeff_win = 0.5;
win_info.overlap_coeff_spectrograms = 0.75;

% win & spectrogram normalization info
SpectrumType = "power"; % "power" "magnitude"
WindowNormalization = false; % true false
FilterBankNormalization = "none"; % "area" "bandwidth" "none"

win_info = extrct_win_info(win_info);

% Create audio feature extractor
afe = audioFeatureExtractor(...
    'SampleRate',       win_info.fs, ...
    'Window',           win_info.win, ...
    'OverlapLength',    win_info.numSamplesOverlapBetweenWin, ...
    'melSpectrum',      true ...
    );

setExtractorParams( ...
    afe,                        'melSpectrum', ...
    'SpectrumType',             SpectrumType, ... % 'power' (default) | 'magnitude'
    'NumBands',                 win_info.num_bands, ...
    "WindowNormalization",      WindowNormalization, ... % true (default) | false
    'FilterBankNormalization',  FilterBankNormalization ... % 'bandwidth' (default) | 'area' | 'none'
    );

% extract auto-segmentation results from yamnet model
% pre-processing - cough event indices
[Features, trueLabels, adsSpecs] = preprocess_classification_CRNN(ads_coughvidNoRaters, win_info, afe);

% check data size before and after pre-processing

num_subj_pre_preprocess = length(ads_coughvidNoRaters.Files);
num_subj_post_preprocess = length(trueLabels);

clc;
disp([num2str(num_subj_pre_preprocess), ' ', num2str(num_subj_post_preprocess)]);

num_seq_per_subj = zeros(length(Features), 1);
for i = 1 : length(Features)
    num_seq_per_subj(i) = size(Features{i}, 4);
end

% num_seq_per_subj = sort(num_seq_per_subj);
% disp(max(num_seq_per_subj));
% disp(Features);

% save data after pre-processing
save('ds.mat', 'Features', 'trueLabels', 'adsSpecs', 'win_info', '-v7.3');

% show time elapsed
clc;
hours = floor(toc/3600);
minuts = floor(toc/60) - hours * 60;
seconds = floor(toc - hours * 3600  - minuts * 60);
disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);

end

