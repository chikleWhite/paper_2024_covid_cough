function [FeaturesCrnn, trueLabels, adsSpecs] = preprocessClassifyCrnn(ads, winInfo, afe, addAugment, coughEventDetect)

% Change current folder
newFolder = "preprocessing-functions";
oldFolder = cd(newFolder);

% % load yamnet model info for auto segmentation
% load('net_info_seg_cough_yamnet.mat', ...
%     'net_seg_cough_yamnet', 'roc_threshold');
% netCoughDetectYamnet = net_seg_cough_yamnet;
% rocThreshold = roc_threshold;

% load yamnet model info for auto segmentation
load('netSegCoughYamnet.mat', ...
    'netSegCoughYamnet', 'rocThreshold');

% pre-processing for segmentation (yamnet network)
[Spectrograms, adsSpecs] = preprocessSegYamnet(ads, winInfo, afe, addAugment, coughEventDetect);

% predict probability for each class
lblsPredDec = predict(netSegCoughYamnet, Spectrograms)';
lblsPredDec = lblsPredDec(2, :);
lblsPredDec = (lblsPredDec >= rocThreshold)';

% erotion & dilation for removal of non-caough leftovers
lblsPredDec = erosionDilationNoiseRemoval(lblsPredDec, winInfo);

% keep only spectrograms that contain cough events:

% pre-allocation
numAudios = size(adsSpecs, 1);
FeaturesCrnn = []; % mel-spectrograms. cell array [96, 64, 1]
trueLabels = []; % true labels (positive/negative)

% loop over all signals and remove spectrograms without cough events
% (loop from last to first)
for i = numAudios : -1 : 1
    
    indx = adsSpecs.audioStartIndx(i) : adsSpecs.audioEndIndx(i);
    
    % choose if to use cough event detection or not
    if coughEventDetect == "no"
        
        % if no spectrograms found -> remove audio from data
        if isempty(indx)
            adsSpecs(i, :) = [];
            continue;
        end
        
        % dont remove segments from cough event detection
        % add features and true lables in opposite direction.
        % suited for RNN model.
        indx = indx(1 : min(length(indx), 150));
        FeaturesCrnn = cat(1, FeaturesCrnn, {Spectrograms(:,:,:,indx)});
        trueLabels = cat(1, trueLabels, adsSpecs.class(i));
        
        % continue to next audio recording
        continue;
    end
    
    nonCoughIndx = indx(lblsPredDec(indx) == 0);
    coughIndx = indx(lblsPredDec(indx) == 1);
    numIndices2remove = length(nonCoughIndx);
    
    % update all end_indx (from i to last subj)
    adsSpecs.audioEndIndx(i : end) = ...
        adsSpecs.audioEndIndx(i : end) - numIndices2remove;
    
    % update all start_indx (from i to last subj)
    if i < numAudios
        adsSpecs.audioStartIndx(i+1 : end) = ...
            adsSpecs.audioStartIndx(i+1 : end) - numIndices2remove;
    end
    
    % if no cough found -> remove audio from data
    if isempty(coughIndx)
        adsSpecs(i, :) = [];
        continue;
    end
    
    % add features and true lables in opposite direction.
    % suited for RNN model.
    FeaturesCrnn = cat(1, FeaturesCrnn, {Spectrograms(:,:,:,coughIndx)});
    trueLabels = cat(1, trueLabels, adsSpecs.class(i));
end

% reverse order back to normal
FeaturesCrnn = FeaturesCrnn(end : -1 : 1);
trueLabels = trueLabels(end : -1 : 1);

% Change the current folder back to the original folder
cd(oldFolder);

end

