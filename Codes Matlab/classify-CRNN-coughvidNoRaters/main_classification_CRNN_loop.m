%% CRNN loop - covid-19 classification - cough audio signals

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpu_selected = "0"; % "0" | "1" | "no"
select_gpu(gpu_selected);

%% run CRNN in loop

% load ds
% [Features, trueLabels, adsSpecs, win_info] = load_ds(); % for 1st time only
load ds;

% shuffle and split data into train val test
shuffle_ds = "yes"; % "yes" | "no" | "TrainVal"
balance_ds = "yes"; % "yes" | "no" | "TrainVal"
split_prcnts = [0.6 0.2 0.2]; % size of train val test in [%]

% load relevent layers from pre-traind CNN network (yamnet/vggish)
net_cnn_name = "vggish"; % "yamnet" | "vggish"
freeze_layers = "no";
lgraph_cnn = load_cnn_lgraph(net_cnn_name, freeze_layers);

% save graph for adding extra layers each iteration
lgraph_yamnet_temp = lgraph_cnn;

% LSTM
numHiddenUnits = 512; % [16 32 64]
drop_prob = 0.1; % 0.05 0.1 0.15 0.2 0.25

MaxEpoch = 30;
miniBatchSize = 4; % [4 8]
InitialLearnRate = 0.0001;
L2Regularization = 0.003;
LearnRateSchedule = "piecewise"; % "none" "piecewise"
LearnRateDropPeriod = 1; % 1 2
LearnRateDropFactor = 0.7;
ValidationPatience = 10; % positive integer | Inf
SequencePaddingDirection = "right"; % "right" "left"
SequenceLength = "longest"; % "longest" (default) | "shortest" | positive integer
checkpointPath = "checkpoint"; % save net in folder after each epoch

tpr_thershold = 0.6;
num_iters = 10;
scores = zeros(3, 9, num_iters);
lbls_pred_vs_true = cell(num_iters, 5);
tic;

for iter = 1 : num_iters
    
    % shuffle and split data into train val test
    [FeaturesTrain, trueLabelsTrain, FeaturesVal, trueLabelsVal, FeaturesTest, trueLabelsTest] = ...
        split_shuffle_ds(Features, trueLabels, shuffle_ds, balance_ds, split_prcnts);
    
    % set classes and classes weights
    uniqueLabels = unique(trueLabelsTrain); % ["negative" "positive"]
    numClasses = length(uniqueLabels); % No. classes
    num_train_pos = sum(trueLabelsTrain == "positive");
    num_train_neg = sum(trueLabelsTrain == "negative");
    num_train = num_train_pos + num_train_neg;
    
    % give weight to each class based on its opposite prevalence
    classWeights = num_train ./ ([num_train_neg num_train_pos] * numClasses);
    
    % create LSTM network
    lstmLayers = [
        bilstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm')
        dropoutLayer(drop_prob, 'Name', 'drop')
        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer("Name", "classification", "Classes", uniqueLabels, "ClassWeights", classWeights)
        ];
    
    % assemble CNN and LSTM networks
    % lstm input -> folding -> CNN -> unfolding -> lstm -> classification
    
    % create empty layer graph
    lgraph = layerGraph;
    
    % add new input layer (sequence input for lstm) and folding layer
    inputSize = [96 64]; % input size needs to be the same as for CNN
    layers = [
        sequenceInputLayer([inputSize 1], 'Name', 'input') % sequence input for lstm
        sequenceFoldingLayer('Name', 'fold') % change structure for CNN
        ];
    
    % connect new input layers to input of CNN
    lgraph = addLayers(lgraph, layers);
    lgraph = addLayers(lgraph, lgraph_cnn.Layers);
    lgraph = connectLayers(lgraph, "fold/out", string(lgraph_cnn.Layers(1, 1).Name));
    
    % add unfolding and lstm layers
    layers = [
        sequenceUnfoldingLayer('Name', 'unfold') % change structure back for lstm
        flattenLayer('Name', 'flatten') % change dim from [1, 1, No. output features] to [No. output features, 1]
        lstmLayers
        ];
    
    % connect unfolding and lstm layers to output of CNN
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, string(lgraph_cnn.Layers(end, 1).Name), "unfold/in");
    
    % add parallel connection between folding & unfolding layers (needed to restore the sequence structure)
    lgraph = connectLayers(lgraph, "fold/miniBatchSize", "unfold/miniBatchSize");
    
    % specify training options
    validationFrequency = floor(numel(trueLabelsTrain)/miniBatchSize);
    options = trainingOptions( ...
        'adam', ...
        'Plots', 'none', ... % 'none' (default) | 'training-progress'
        'Verbose', false, ... % 1 (true) (default) | 0 (false)
        'MaxEpochs', MaxEpoch, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ... % 'once' (default) | 'never' | 'every-epoch'
        ...
        'ValidationData', {FeaturesVal, trueLabelsVal}, ...
        'ValidationFrequency', validationFrequency, ...
        'InitialLearnRate', InitialLearnRate, ...
        'LearnRateSchedule', LearnRateSchedule, ... % 'none' (default) | 'piecewise'
        'LearnRateDropPeriod', LearnRateDropPeriod, ...
        'LearnRateDropFactor', LearnRateDropFactor, ...
        'L2Regularization', L2Regularization, ...
        'ValidationPatience', ValidationPatience, ...
        ...
        'SequencePaddingDirection', SequencePaddingDirection, ...
        'SequenceLength', SequenceLength, ...% "longest" (default) | "shortest" | positive integer
        ...
        'CheckpointPath', checkpointPath, ...
        'BatchNormalizationStatistics', 'moving' ... % 'population' (default) | 'moving'
        );
    
    % remove saved nets from previous training
    delete(checkpointPath + "/*");
    
    % train network
    [~, net_info] = trainNetwork(FeaturesTrain, trueLabelsTrain, lgraph, options);
    
    % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
    net_classification_CRNN = extract_net_from_checkpoint(net_info, checkpointPath);
    
    % remove saved nets after training
    delete(checkpointPath + "/*");
    
    % predict on train, val and test sets (probability [0,1])
    miniBatchSize_predict = 1;
    positive_indx = find(string(uniqueLabels) == "positive");
    predLabelsTrain = predict(net_classification_CRNN, FeaturesTrain, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsTrain = predLabelsTrain(:, positive_indx);
    predLabelsVal   = predict(net_classification_CRNN, FeaturesVal, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsVal   = predLabelsVal(:, positive_indx);
    predLabelsTest  = predict(net_classification_CRNN, FeaturesTest, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsTest  = predLabelsTest(:, positive_indx);
    
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
    % tpr = 1, fpr = 0 & sensitivity >= 0.6
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
        auc = CalculateEmpiricalAUC(lbls_pred_cell{i}, string(lbls_true_cell{i}) == "positive");
        
        [val_loss, val_loss_indx] = min(net_info.ValidationLoss);
        train_loss = ...
            mean(net_info.TrainingLoss(max(val_loss_indx - validationFrequency + 1, 1) : val_loss_indx));
        
        % put values in matrix
        scores(i, :, iter) = [acc, uar, F1, sensitivity, PPV, specificity, auc, val_loss, train_loss];
    end
    
    % average scores
    scores_avg = ...
        string(round(mean(scores(:, :, 1 : iter), 3), 3) * 100) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scores(:, :, 1 : iter), 0, 3), 3) * 100);
    
    % save results & predictions in mat file
    save scores_loop scores scores_avg lbls_pred_vs_true
    
    clc; % remove previous text from command window
    
    % display scores
    scores_names = ["accuracy" "uar" "F1" "sensitivity" "PPV" "specificity" "auc" "val_loss" "train_loss"];
    disp(scores_names);
    disp(scores(:, :, 1 : iter));
    disp([scores_names ; scores_avg]);
    
    % display progress done and time
    dispProgressAndTime(num_iters, iter);
end

% deselects the GPU device and clears its memory
gpuDevice([]);
