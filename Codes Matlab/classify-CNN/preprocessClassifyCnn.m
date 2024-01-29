function [FeaturesCNN, trueLabels, adsSpecs] = preprocessClassifyCnn(ADS, winInfo, afe, coughEventDetect)

% Change current folder
newFolder = "preprocessing-functions";
oldFolder = cd(newFolder);

% load yamnet model info for auto segmentation
load('netSegCoughYamnet.mat', ...
    'netSegCoughYamnet', 'rocThreshold');

% pre-processing for segmentation (yamnet network)
[Spectrograms, adsSpecs] = preprocessSegYamnet(ADS, winInfo, afe, coughEventDetect);

% predict probability for each class
lbls_pred_dec = predict(netSegCoughYamnet, Spectrograms)';
lbls_pred_dec = lbls_pred_dec(2, :);
lbls_pred_dec = (lbls_pred_dec >= rocThreshold)';

% erotion & dilation for removal of non-caough leftovers
lbls_pred_dec = erosion_dilation_noise_removal(lbls_pred_dec, winInfo);

% keep only spectrograms that contain cough events

% pre-allocation
numAudios = size(adsSpecs, 1);
FeaturesCNN = []; % mel-spectrograms. cell array [96, 64, 1]
trueLabels = []; % true labels (positive/negative)

% choose if to use cough event detection or not
if coughEventDetect == "no"
    
    % loop over all signals and remove spectrograms without cough events
    % (loop from last to first)
    for i = numAudios : -1 : 1
        
        indx = adsSpecs.audioStartIndx(i) : adsSpecs.audioEndIndx(i);
        
        % if no spectrograms found -> remove audio from data
        if isempty(indx)
            adsSpecs(i, :) = [];
            continue;
        end
        
        % dont remove segments from cough event detection
        % add features and true lables in opposite direction.
        % suited for cnn model.
        FeaturesCNN = cat(4, FeaturesCNN, Spectrograms(:,:,:,indx));
        trueLabels = cat(1, trueLabels, repmat(adsSpecs.class(i), length(indx), 1));
    end
    
else % use cough event detection
    
    % loop over all signals and remove spectrograms without cough events
    % (loop from last to first)
    for i = numAudios : -1 : 1
        
        indx = adsSpecs.audioStartIndx(i) : adsSpecs.audioEndIndx(i);
        nonCoughIndx = indx(lbls_pred_dec(indx) == 0);
        coughIndx = indx(lbls_pred_dec(indx) == 1);
        numIndices2remove = length(nonCoughIndx);
        
        % update all end_indx (from i to last subj)
        adsSpecs.audioEndIndx(i : end) = ...
            adsSpecs.audioEndIndx(i : end) - numIndices2remove;
        
        % update all start_indx (from i to last subj)
        if i < numAudios
            adsSpecs.audioStartIndx(i+1 : end) = ...
                adsSpecs.audioStartIndx(i+1 : end) - numIndices2remove;
        end
        
        % if no cough found -> remove subject from data
        if isempty(coughIndx)
            adsSpecs(i, :) = [];
            continue;
        end
        
        % add features and true lables in opposite direction.
        % suited for cnn model.
        FeaturesCNN = cat(4, FeaturesCNN, Spectrograms(:,:,:,coughIndx));
        trueLabels = cat(1, trueLabels, repmat(adsSpecs.class(i), length(coughIndx), 1));
    end
end

% reverse order back to normal
FeaturesCNN = flip(FeaturesCNN, 4);
trueLabels = flip(trueLabels, 1);

% Change the current folder back to the original folder
cd(oldFolder);

end

