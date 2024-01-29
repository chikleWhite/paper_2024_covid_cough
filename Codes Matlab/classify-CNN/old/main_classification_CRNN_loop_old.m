%% Covid classification - CRNN

% remove previous data
close all; clc; clear;

% load VGGish neural network for features embedding
downloadFolder = fullfile(tempdir,'VGGishDownload');
loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/vggish.zip');
VGGishLocation = tempdir;
unzip(loc,VGGishLocation);
addpath(fullfile(VGGishLocation,'vggish'));
net_cnn = vggish;

% roc_threshold = 0.72; % 0.5 0.72 0.82
tpr_thershold = 0.6;
num_iters = 100; % number of iterations
scores = zeros(3, 6, num_iters);
lbls_pred_vs_true = cell(num_iters, 5);
tic;

for iter = 1 : num_iters
    
    % Create audioDatastore objects of the data and split it into train, validation and test sets
    dataset_name = "all"; % choose dataset
    shuffle_dataset = "yes"; % choose if to shuffle the datasets
    
    % load data
    [adsTrain, adsVal, adsTest] = load_datasets_train_val_test(shuffle_dataset, dataset_name);
    
    % choose hyper-parameters
    % win & spectrogram length/overlap info
    win_info.win_len = 128;
    win_info.overlap_coeff_win = 0.5;
    win_info.overlap_coeff_spectrograms = 0.75;
    
    % win & spectrogram normalization info
    SpectrumType = "power"; % "power" "magnitude"
    WindowNormalization = false; % true false
    FilterBankNormalization = "none"; % "area" "bandwidth" "none"
    
    % LSTM
    numHiddenUnits = 512;
    drop_prob = 0.1; % 0.05 0.1 0.15 0.2 0.25
    
    MaxEpochs = 30;
    miniBatchSize = 4;
    InitialLearnRate = 0.0001;
    L2Regularization = 0.003;
    LearnRateSchedule = "piecewise"; % "none" "piecewise"
    LearnRateDropPeriod = 1; % 1 2
    LearnRateDropFactor = 0.7;
    ValidationPatience = 5;
    SequencePaddingDirection = "right"; % "right" "left"
    SequenceLength = "longest"; % "longest" (default) | "shortest" | positive integer
    
    win_info = extrct_win_info(win_info);
    
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
    
    % preprocess datasets: extract auto-segmentation results from yamnet model
    % pre-processing - cough event indices
    [FeaturesTrain, trueLabelsTrain, adsSpecsTrain] = preprocess_classification_CRNN(adsTrain, win_info, afe);
    [FeaturesVal, trueLabelsVal, adsSpecsVal] = preprocess_classification_CRNN(adsVal, win_info, afe);
    [FeaturesTest, trueLabelsTest, adsSpecsTest] = preprocess_classification_CRNN(adsTest, win_info, afe);
    
    %     % Sort Sequences by Length
    %     sequenceLengths = cellfun(@(X) size(X,4), FeaturesTrain);
    %     [sequenceLengthsSorted, idx] = sort(sequenceLengths);
    %     FeaturesTrain = FeaturesTrain(idx);
    %     trueLabelsTrain = trueLabelsTrain(idx);
    %
    %     % Truncate Sequences: reduce length of sequences to max size
    %     max_seq_len = 80;
    %     idxSorted = 1 : length(sequenceLengthsSorted);
    %     idx_above_th = idxSorted(sequenceLengthsSorted > max_seq_len);
    %     for subj_indx = idx_above_th
    %         sequence = FeaturesTrain{subj_indx, 1};
    %         sequence = sequence(:,:,:, 1 : max_seq_len);
    %         FeaturesTrain{subj_indx, 1} = sequence;
    %     end
    
    % set classes and classes weights
    uniqueLabels = unique(trueLabelsTrain);
    numLabels = 2;
    num_train_pos = sum(trueLabelsTrain == "positive");
    num_train_neg = sum(trueLabelsTrain == "negative");
    num_train = num_train_pos + num_train_neg;
    classWeightsUniform = 'none';
    classWeightsNonUniform = [num_train_pos, num_train_neg] / num_train;
    
    classWeights = classWeightsNonUniform;
    
    % create LSTM network
    numFeatures = net_cnn.Layers(22, 1).OutputSize; % No. output features from CNN
    numClasses = 2; % No. classes
    
    lstmLayers = [
        bilstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm')
        dropoutLayer(drop_prob, 'Name', 'drop')
        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer("Name", "classification", "Classes", uniqueLabels, "ClassWeights", classWeights)
        ];
    
    % assemble CNN and LSTM networks
    % lstm input -> folding -> CNN -> unfolding -> lstm -> classification
    cnnLayers = layerGraph(net_cnn.Layers);
    
    % remove layers from CNN: 1st (input) and last (regression)
    layerNames = [string(net_cnn.Layers(1, 1).Name), string(net_cnn.Layers(24, 1).Name)];
    cnnLayers = removeLayers(cnnLayers, layerNames);
    
    % add new input layer (sequence input for lstm) and folding layer
    inputSize = net_cnn.Layers(1).InputSize(1:2); % input size needs to be the same as for CNN
    layers = [
        sequenceInputLayer([inputSize 1], 'Name', 'input') % sequence input for lstm
        sequenceFoldingLayer('Name', 'fold') % change structure for CNN
        ];
    
    % connect new input layers to input of CNN
    lgraph = addLayers(cnnLayers, layers);
    lgraph = connectLayers(lgraph, "fold/out", string(net_cnn.Layers(2, 1).Name));
    
    % add unfolding and lstm layers
    layers = [
        sequenceUnfoldingLayer('Name', 'unfold') % change structure back for lstm
        flattenLayer('Name', 'flatten') % change dim from [1,1,128] to [128]
        lstmLayers
        ];
    
    % connect unfolding and lstm layers to output of CNN
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, string(net_cnn.Layers(23, 1).Name), "unfold/in");
    
    % add parallel connection between folding & unfolding layers (needed to restore the sequence structure)
    lgraph = connectLayers(lgraph, "fold/miniBatchSize", "unfold/miniBatchSize");
    
    % specify training options
    numObservations = numel(FeaturesTrain);
    validationFrequency = floor(numObservations / miniBatchSize);
    
    options = trainingOptions( ...
        'adam', ...
        'MaxEpochs', MaxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'Verbose', false, ... % 1 (true) (default) | 0 (false)
        'Shuffle', 'every-epoch', ... % 'once' (default) | 'never' | 'every-epoch'
        'ValidationData', {FeaturesVal, trueLabelsVal}, ...
        'ValidationFrequency', validationFrequency, ...
        'InitialLearnRate', InitialLearnRate, ...
        'LearnRateSchedule', 'piecewise', ... % piecewise, none
        'LearnRateDropPeriod', 1, ...
        'LearnRateDropFactor', LearnRateDropFactor, ...
        'L2Regularization', L2Regularization, ...
        'SequencePaddingDirection', SequencePaddingDirection, ...
        'SequenceLength', SequenceLength, ... % "longest" (default) | "shortest" | positive integer
        'ValidationPatience', ValidationPatience, ...
        'Plots', 'none' ... % 'none' (default) | 'training-progress'
        );
    
    % Train the LSTM network with the specified training options.
    net_classification_CRNN = trainNetwork(FeaturesTrain, trueLabelsTrain, lgraph, options);
    
    % predict on train, val and test sets (probability [0,1])
    miniBatchSize_predict = 1;
    predLabelsTrain = predict(net_classification_CRNN, FeaturesTrain, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsTrain = predLabelsTrain(:, 2);
    predLabelsVal   = predict(net_classification_CRNN, FeaturesVal, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsVal   = predLabelsVal(:, 2);
    predLabelsTest  = predict(net_classification_CRNN, FeaturesTest, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsTest  = predLabelsTest(:, 2);
    
    % find best ROC thrshold on val set, and plot ROC curve
    % plot ROC curve and confusion matrix on test set.
    % display for train, val and test sets:
    
    lbls_pred_cell = {predLabelsTrain predLabelsVal predLabelsTest};
    lbls_true_cell = {trueLabelsTrain trueLabelsVal trueLabelsTest};
    
    % get ROC of val set to find best threshold
    
    % change true labels from name of classes to logical (0/1) - on val set
    lbls_true = (string(lbls_true_cell{2}) == "positive")';
    
    % pred labels from val set
    lbls_pred = lbls_pred_cell{2}';
    
    % get ROC info
    [tpr, fpr, roc_thresholds] = roc(lbls_true, lbls_pred); % ROC values (TPR = sensetivity, FPR = 1 - specificity)
    
    % save predictions & true lables
    lbls_pred_vs_true(iter, :) = {lbls_pred_cell, lbls_true_cell, tpr, fpr, roc_thresholds};
    
    % find optimal threshold based on minimum distance from point (1,0) ->
    % tpr = 1, fpr = 0 & sensitivity >= 0.5
    indx_low_sensitivity = tpr < tpr_thershold;
    dist_from_opt_point = ((1 - tpr).^2 + (0 - fpr).^2).^0.5;
    dist_from_opt_point(indx_low_sensitivity) = 1;
    [~, indx_min_dist] = min(dist_from_opt_point);
    roc_threshold = roc_thresholds(indx_min_dist);
    
    % loop over train, val and test sets. plot scores on all and add confusion
    % matrix on test set only
    for i = 1 : length(lbls_pred_cell)
        
        % convert predictions from probabilities to logical (0/1)
        lbls_pred = (lbls_pred_cell{i} >= roc_threshold);
        
        % convert predictions from logical (0/1) to categorical (names of classes)
        lbls_pred = categorical(lbls_pred, [0 1], {'negative' 'positive'});
        
        % convert labels (predictions and true) from categorical to string
        lbls_pred = string(lbls_pred);
        lbls_true = string(lbls_true_cell{i});
        
        % TP TN FP FN
        TP = sum(lbls_pred == lbls_true & lbls_true == "positive");
        TN = sum(lbls_pred == lbls_true & lbls_true == "negative");
        FP = sum(lbls_pred ~= lbls_true & lbls_true == "negative");
        FN = sum(lbls_pred ~= lbls_true & lbls_true == "positive");
        
        % calculate scores per segment
        acc         =   (TP + TN) / (TP + TN + FP + FN);
        uar         =   (TP / (TP + FN) + TN / (TN + FP)) / 2;
        F1          =   TP ./ (TP + (FP + FN)/2);
        sensitivity =   TP / (TP + FN);
        PPV         =   TP / (TP + FP);
        specificity =   TN / (TN + FP);
        
        % put values in matrix
        scores(i, :, iter) = [acc, uar, F1, sensitivity, PPV, specificity];
    end
    
    scores_avg = ...
        string(round(mean(scores(:, :, 1 : iter), 3), 3) * 100) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scores(:, :, 1 : iter), 0, 3), 3) * 100);
    
    clc; % remove previous text from command window
    
    % show scores
    disp(scores(:, :, 1 : iter));
    disp(scores_avg);
    
    % show current iter number and percent done
    disp(['work done: ', num2str(iter), ' iters out of ', num2str(num_iters), ' (', num2str(round(iter / num_iters * 100, 2)), ' %)']);
    
    % show time elapsed
    hours = floor(toc/3600);
    minuts = floor(toc/60) - hours * 60;
    seconds = floor(toc - hours * 3600  - minuts * 60);
    disp(['time elapsed: ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);
    
    % show time remain
    time_remain = toc / iter * (num_iters - iter);
    hours = floor(time_remain/3600);
    minuts = floor(time_remain/60) - hours * 60;
    seconds = floor(time_remain - hours * 3600  - minuts * 60);
    disp(['time remain ~ ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);
end

save scores_loop scores scores_avg lbls_pred_vs_true

% roc_threshold = 0.85;
% lbls_pred = (predLabelsTest >= roc_threshold);
% lbls_true = (string(trueLabelsTest) == "positive");
% sequenceLengths = cellfun(@(X) size(X,4), FeaturesTrain);
% comp_results_per_seq_len = [lbls_pred == lbls_true, sequenceLengths];
% [sequenceLengthsSorted, idx] = sort(sequenceLengths);
% comp_results_per_seq_lenSorted = comp_results_per_seq_len(idx, :);
