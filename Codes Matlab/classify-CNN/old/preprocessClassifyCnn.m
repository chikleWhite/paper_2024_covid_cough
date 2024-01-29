function [FeaturesCNN, trueLabels, adsSpecs] = preprocessClassifyCnn(ADS, winInfo, afe)

% Change current folder
newFolder = "preprocessing-functions";
oldFolder = cd(newFolder);

% load yamnet model info for auto segmentation
load('net_info_seg_cough_yamnet.mat', ...
    'net_seg_cough_yamnet', 'roc_threshold');

% pre-processing for segmentation (yamnet network)
[FeaturesCNN_before_seg, adsSpecs] = preprocess_seg_yamnet(ADS, winInfo, afe);

% predict probability for each class
lbls_pred_dec = predict(net_seg_cough_yamnet, FeaturesCNN_before_seg)';
lbls_pred_dec = lbls_pred_dec(2, :);
lbls_pred_dec = (lbls_pred_dec >= roc_threshold)';

% erotion & dilation for removal of non-caough leftovers
lbls_pred_dec = erosion_dilation_noise_removal(lbls_pred_dec, winInfo);

% keep only spectrograms that contain cough events

% pre-allocation
num_sig = size(adsSpecs, 1);
FeaturesCNN = []; % mel-spectrograms. cell array [96, 64, 1]
trueLabels = []; % true labels (positive/negative)

% loop over all signals and remove spectrograms without cough events
% (loop from last to first)
for i = num_sig : -1 : 1
    
    indx = adsSpecs.start_indx(i) : adsSpecs.end_indx(i);
    
    non_cough_indx = indx(lbls_pred_dec(indx) == 0);
    cough_indx = indx(lbls_pred_dec(indx) == 1);
    num_indices2remove = length(non_cough_indx);
    
    % update all end_indx (from i to last subj)
    adsSpecs.end_indx(i : end) = ...
        adsSpecs.end_indx(i : end) - num_indices2remove;
    
    % update all start_indx (from i to last subj)
    if i < num_sig
        adsSpecs.start_indx(i+1 : end) = ...
            adsSpecs.start_indx(i+1 : end) - num_indices2remove;
    end
    
    % if no cough found -> remove subject from data
    if isempty(cough_indx)
        adsSpecs(i, :) = [];
        continue;
    end
    
    % add features and true lables in opposite direction.
    % suited for cnn model.
    FeaturesCNN = cat(4, FeaturesCNN, FeaturesCNN_before_seg(:,:,:,cough_indx));
    trueLabels = cat(1, trueLabels, repmat(adsSpecs.label(i), length(cough_indx), 1));
end

% reverse order back to normal
FeaturesCNN = flip(FeaturesCNN, 4);
trueLabels = flip(trueLabels, 1);

% Change the current folder back to the original folder
cd(oldFolder);

end

