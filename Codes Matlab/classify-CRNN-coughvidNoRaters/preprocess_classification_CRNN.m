function [FeaturesCRNN, trueLabels, adsSpecs] = preprocess_classification_CRNN(ADS, win_info, afe)

% Change current folder
newFolder = "preprocessing-functions";
oldFolder = cd(newFolder);

% load yamnet model info for auto segmentation
load('net_info_seg_cough_yamnet.mat', ...
    'net_seg_cough_yamnet', 'roc_threshold');

% pre-processing for segmentation (yamnet network)
disp('pre-processing for segmentation (yamnet network)');
[FeaturesCNN, adsSpecs] = preprocess_seg_yamnet(ADS, win_info, afe);

% predict probability for each class
lbls_pred_dec = predict(net_seg_cough_yamnet, FeaturesCNN)';
lbls_pred_dec = lbls_pred_dec(2, :);
lbls_pred_dec = (lbls_pred_dec >= roc_threshold)';

% erotion & dilation for removal of non-caough leftovers
lbls_pred_dec = erosion_dilation_noise_removal(lbls_pred_dec, win_info);

% keep only spectrograms that contain cough events

% pre-allocation
num_sig = size(adsSpecs, 1);
FeaturesCRNN = []; % mel-spectrograms. cell array [96, 64, 1]
trueLabels = []; % true labels (positive/negative)

% loop over all signals and remove spectrograms without cough events
% (loop from last to first)
disp('remove spectrograms without cough events');
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
    % suited for RNN model.
    FeaturesCRNN = cat(1, FeaturesCRNN, {FeaturesCNN(:,:,:,cough_indx)});
    trueLabels = cat(1, trueLabels, adsSpecs.label(i));
    
    % print progress done
    prcnt_done = round((num_sig - i) / num_sig * 100);
    fprintf([repmat('\b', 1, 5), ' %d%%'], prcnt_done);
end

% reverse order back to normal
FeaturesCRNN = FeaturesCRNN(end : -1 : 1);
trueLabels = trueLabels(end : -1 : 1);

% Change the current folder back to the original folder
cd(oldFolder);

end

