function [Features, trueLabels, adsSpecs] = preprocess_classification_lstm(ADS)

% Change current folder
newFolder = "preprocessing-lstm-functions";
oldFolder = cd(newFolder);

% load yamnet model info for auto segmentation
load('net_info_seg_cough_yamnet.mat', ...
    'net_seg_cough_yamnet', 'win_info', 'roc_threshold');

% pre-processing for segmentation (yamnet network)
[Features, adsSpecs] = preprocess_seg_yamnet(ADS, win_info);

disp('features extraction for segmentation done!!');

% predict probability for each class
lbls_pred_dec = predict(net_seg_cough_yamnet, Features)';
lbls_pred_dec = lbls_pred_dec(2, :);
lbls_pred_dec = (lbls_pred_dec >= roc_threshold)';

% erotion & dilation for removal of non-cough leftovers
len_records = size(adsSpecs, 1);
for i = 1 : len_records
    indx_range = (adsSpecs.start_indx(i) : adsSpecs.end_indx(i));
    lbls_pred_dec(indx_range) = erosion_dilation_noise_removal(lbls_pred_dec(indx_range));
end

% convert prediction results to segmentation indices
coughEventLocations = ...
    coughIndx2eventLocation(ADS, lbls_pred_dec, adsSpecs, win_info);

disp('segmentation done!!');

% extract features from each frame as input to network
[Features, trueLabels, adsSpecs] = extract_features_lpcc(ADS, coughEventLocations);

% % turn categorical labels to double (0/1)
% trueLabels(trueLabels == categorical("positive")) = categorical(1);
% trueLabels(trueLabels == categorical("negative")) = categorical(2);
% trueLabels = removecats(trueLabels); % remove unused categories
% trueLabels = double(trueLabels) - 1;

disp('features extraction for classification done!!');

% Change the current folder back to the original folder
cd(oldFolder);

end

