%% CRNN loop - covid-19 classification - cough audio signals

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpu_selected = "0"; % "0" | "1" | "no"
select_gpu(gpu_selected);

%% run CRNN in loop

% datasets loading and spliting info
dataset_name = "all_ds"; % "all_ds" | "voca" | "voca_old" | "coughvid"
split_prcnts = [0.6 0.2 0.2]; % size of train val test in [%]
shuffle_datasets = "all"; % "all" | "train_val" | "no"

% load relevent layers from pre-traind CNN network (yamnet/vggish)
net_cnn_name = "vggish"; % "yamnet" | "vggish" | openl3
freeze_layers = "no";
lgraph_cnn = load_cnn_lgraph(net_cnn_name, freeze_layers);

% win & spectrogram length/overlap info
winInfo.winLen = 128;
winInfo.overlapCoeffWin = 0.5;
winInfo.overlapCoeffSpectrograms = 0.8; % [0.75 0.8 0.85 0.9]

% win & spectrogram normalization info
winInfo.SpectrumType = "power"; % "power" | "magnitude"
winInfo.WindowNormalization = false; % true | false
winInfo.FilterBankNormalization = "none"; % "area" | "bandwidth" | "none"

hyperParameters.MaxEpoch = 30;
hyperParameters.miniBatchSize = 16; % [4 8]
hyperParameters.InitialLearnRate = 0.0003;
hyperParameters.L2Regularization = 0.003;
hyperParameters.LearnRateSchedule = "piecewise"; % "none" "piecewise"
hyperParameters.LearnRateDropPeriod = 1; % 1 2
hyperParameters.LearnRateDropFactor = 0.7;
hyperParameters.ValidationPatience = 5; % positive integer | Inf
hyperParameters.checkpointPath = "checkpoint"; % save net in folder after each epoch

tprThershold = 0.6;
numIters = 100;
scoresMat = zeros(3, 8, numIters);
lblsPredTrue = cell(numIters, 2);
tic;

for iter = 1 : numIters
    
    % Streamline audio feature extraction (mel-spectrogram)
    [afe, winInfo] = setAudioFeatureExtractor(winInfo);
    
    % load data and split to train val test
    [adsTrain, adsVal, adsTest] = load_split_shuffle_datasets(dataset_name, split_prcnts, shuffle_datasets);
    
    % preprocess datasets: extract auto-segmentation results from yamnet model
    [FeaturesPerSegTrain, trueLabelsPerSegTrain, adsSpecsTrain] = ...
        preprocess_classification_CNN(adsTrain, winInfo, afe);
    [FeaturesPerSegVal, trueLabelsPerSegVal, adsSpecsVal] = ...
        preprocess_classification_CNN(adsVal, winInfo, afe);
    [FeaturesPerSegTest, trueLabelsPerSegTest, adsSpecsTest] = ...
        preprocess_classification_CNN(adsTest, winInfo, afe);
    
    % set classes and classes weights
    uniqueLabels = unique(trueLabelsPerSegVal); % ["negative" "positive"]
    numClasses = length(uniqueLabels); % No. classes
    num_train_pos = sum(trueLabelsPerSegTrain == "positive");
    num_train_neg = sum(trueLabelsPerSegTrain == "negative");
    num_train = num_train_pos + num_train_neg;
    
    % give weight to each class by the other one relative size
    classWeightsNonUniform = num_train ./ ([num_train_neg num_train_pos] * numClasses);
    classWeights = classWeightsNonUniform;
    
    % assemble CNN network
    % correct image input layer -> loaded cnn -> last fully connected layer + relu/max pooling -> softmax -> classification
    % create empty layer graph
    lgraph = layerGraph;
    
    % add new input layer (image input layer to cnn)
    inputSize = [96 64 1]; % input size needs to be the same as for CNN
    inputlayer = imageInputLayer(inputSize,'Name','input');
    
    % connect new input layer to cnn
    lgraph = addLayers(lgraph, inputlayer);
    lgraph = addLayers(lgraph, lgraph_cnn.Layers);
    lgraph = connectLayers(lgraph, "input", string(lgraph_cnn.Layers(1, 1).Name));
    
    % add last layers (fully connected, softmax, classification)
    lastLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer("Name", "classification", "Classes", uniqueLabels, "ClassWeights", classWeights)
        ];
    
    % connect last layers to cnn
    lgraph = addLayers(lgraph, lastLayers);
    lgraph = connectLayers(lgraph, string(lgraph_cnn.Layers(end, 1).Name), lastLayers(1, 1).Name);
    
    % specify training options
    validationFrequency = floor(size(FeaturesPerSegTrain, 4) / hyperParameters.miniBatchSize);
    options = trainingOptions( ...
        'adam', ...
        'Plots', 'none', ... % 'none' (default) | 'training-progress'
        'Verbose', false, ... % 1 (true) (default) | 0 (false)
        'MaxEpochs', hyperParameters.MaxEpoch, ...
        'MiniBatchSize', hyperParameters.miniBatchSize, ...
        'Shuffle', 'every-epoch', ... % 'once' (default) | 'never' | 'every-epoch'
        ...
        'ValidationData', {FeaturesPerSegVal, trueLabelsPerSegVal}, ...
        'ValidationFrequency', validationFrequency, ...
        'InitialLearnRate', hyperParameters.InitialLearnRate, ...
        'LearnRateSchedule', hyperParameters.LearnRateSchedule, ... % 'none' (default) | 'piecewise'
        'LearnRateDropPeriod', hyperParameters.LearnRateDropPeriod, ...
        'LearnRateDropFactor', hyperParameters.LearnRateDropFactor, ...
        'L2Regularization', hyperParameters.L2Regularization, ...
        'ValidationPatience', hyperParameters.ValidationPatience, ...
        ...
        'CheckpointPath', hyperParameters.checkpointPath, ...
        'BatchNormalizationStatistics', 'moving' ... % 'population' (default) | 'moving'
        );
    
    % remove saved nets from previous training
    delete(hyperParameters.checkpointPath + "/*");
    
    % train network
    [~, net_info] = trainNetwork(FeaturesPerSegTrain, trueLabelsPerSegTrain, lgraph, options);
    
    % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
    net_classification_CNN = extract_net_from_checkpoint(net_info, hyperParameters.checkpointPath);
    
    % remove saved nets after training
    delete(hyperParameters.checkpointPath + "/*");
    
    % predict on validation set
    positive_indx = find(string(uniqueLabels) == "positive");
    predLabelsPerSegTrain = predict(net_classification_CNN, FeaturesPerSegTrain, 'MiniBatchSize', hyperParameters.miniBatchSize);
    predLabelsPerSegTrain = predLabelsPerSegTrain(:, positive_indx);
    predLabelsPerSegVal   = predict(net_classification_CNN, FeaturesPerSegVal, 'MiniBatchSize', hyperParameters.miniBatchSize);
    predLabelsPerSegVal   = predLabelsPerSegVal(:, positive_indx);
    predLabelsPerSegTest  = predict(net_classification_CNN, FeaturesPerSegTest, 'MiniBatchSize', hyperParameters.miniBatchSize);
    predLabelsPerSegTest  = predLabelsPerSegTest(:, positive_indx);
    
    % change predictions and true labels from per spectrogram to per subject
    [predLabelsPerSubjTrain, trueLabelsPerSubjTrain] = perSegment2perSubject(predLabelsPerSegTrain, trueLabelsPerSegTrain, adsSpecsTrain);
    [predLabelsPerSubjVal, trueLabelsPerSubjVal] = perSegment2perSubject(predLabelsPerSegVal, trueLabelsPerSegVal, adsSpecsVal);
    [predLabelsPerSubjTest, trueLabelsPerSubjTest] = perSegment2perSubject(predLabelsPerSegTest, trueLabelsPerSegTest, adsSpecsTest);
    
    % gather results in cells
    lblsPredCell = {predLabelsPerSubjTrain predLabelsPerSubjVal predLabelsPerSubjTest};
    lblsTrueCell = {trueLabelsPerSubjTrain trueLabelsPerSubjVal trueLabelsPerSubjTest};
    
    % save predictions & true lables
    lblsPredTrue(iter, :) = {lblsPredCell, lblsTrueCell};
    
    % find best ROC threshold for val set
    setRocThreshold = "auto";
    plotRoc = "no";
    rocThreshold = findBestThresholdAndPlotROC(lblsPredCell{2}, lblsTrueCell{2}, ...
        tprThershold, setRocThreshold, plotRoc);
    
    % loop over train, val and test sets. plot scores on all and add confusion
    % matrix on test set only
    for i = 1 : length(lblsPredCell)
        
        % calculate scores
        plotConfusionMat = "no";
        scoresStruct = calculateScores(lblsPredCell{i}, lblsTrueCell{i}, rocThreshold, plotConfusionMat);
        
        % put values in matrix
        scoresMat(i, 1 : end - 1, iter) = [...
            scoresStruct.acc, ...
            scoresStruct.uar, ...
            scoresStruct.F1, ...
            scoresStruct.sensitivity, ...
            scoresStruct.PPV, ...
            scoresStruct.specificity, ...
            scoresStruct.auc];
    end
    
    % calculate train and val losses
    [val_loss, val_loss_indx] = min(net_info.ValidationLoss);
        train_loss = ...
            mean(net_info.TrainingLoss(max(val_loss_indx - validationFrequency + 1, 1) : val_loss_indx));
   
    % add losses to scoresMat
    scoresMat(:, end, iter) = [train_loss ; val_loss ; nan];
    
    % average scores
    scores_avg = ...
        string(round(mean(scoresMat(:, :, 1 : iter), 3), 1)) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scoresMat(:, :, 1 : iter), 0, 3), 1));
    
    % save results & predictions in mat file
    save scores_loop scoresMat scores_avg lblsPredTrue
    clc; % remove previous text from command window
    
    % display scores
    scores_names = ["accuracy" "uar" "F1" "sensitivity" "PPV" "specificity" "auc" "loss"];
    disp('scores:');
    disp(scores_names);
    disp(scoresMat(:, :, 1 : iter));
    disp('average scores:');
    disp([scores_names ; scores_avg]);
    
    % display progress done and time
    dispProgressAndTime(numIters, iter);
end

% deselects the GPU device and clears its memory
gpuDevice([]);
