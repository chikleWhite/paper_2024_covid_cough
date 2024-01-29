%% CRNN loop - covid-19 classification - cough audio signals

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpuSelected = "0"; % "0" | "1" | "no"
selectGpu(gpuSelected);

%% run CRNN in loop

% iterations options
generalInfo.numIters = 10;
% hyperParameters.rocThreshold = 0.6;
hyperParameters.specificityThershold = 0.65;
generalInfo.coughEventDetect = "yes"; % use cough event detectio? "yes" | "no"

% measure testing time
timeTest = zeros(generalInfo.numIters, 1);

% datasets loading and spliting info
generalInfo.datasetNames = ["voca" "coughvid" "coswara"]; % ["voca" "coughvid" "coswara"]
generalInfo.splitPrcnts = [0.6 0.2 0.2]; % size of train val test in [%]
generalInfo.shuffleDatasets = "all"; % "all" | "trainVal" | "no"
generalInfo.changeSubjectsOrder = "no"; % "yes" | "no"
generalInfo.RandomSeed = 1; % 1 | "shuffle"
rng(generalInfo.RandomSeed);

% load relevent layers from pre-traind CNN network (yamnet/vggish)
netCnnName = "vggish"; % "yamnet" | "vggish"
freezeLayers = "no";
lgraphCnn = loadCnnLgraph(netCnnName, freezeLayers);

% win & spectrogram length/overlap info
winInfo.winLen = 128;
winInfo.overlapCoeffWin = 0.5;
winInfo.overlapCoeffSpectrograms = 0.8; % [0.75 0.8 0.85 0.9]

% win & spectrogram normalization info
winInfo.SpectrumType = "magnitude"; % "power" | "magnitude"
winInfo.WindowNormalization = false; % true | false
winInfo.FilterBankNormalization = "none"; % "area" | "bandwidth" | "none"

% LSTM
hyperParameters.numHiddenUnits = 512; % [16 32 64]
hyperParameters.dropProb = 0.3; % 0.05 0.1 0.15 0.2 0.25

% training options
hyperParameters.MaxEpochs = 30;
hyperParameters.miniBatchSize = 4; % [4 8]
hyperParameters.InitialLearnRate = 0.00003;
hyperParameters.L2Regularization = 0.01;
hyperParameters.LearnRateSchedule = "piecewise"; % "none" | "piecewise"
hyperParameters.LearnRateDropPeriod = 1; % 1 2
hyperParameters.LearnRateDropFactor = 0.99;
hyperParameters.ValidationPatience = 10; % positive integer | Inf
hyperParameters.SequencePaddingDirection = "right"; % "right" | "left"
hyperParameters.SequenceLength = "longest"; % "longest" (default) | "shortest" | positive integer
hyperParameters.checkpointPath = "checkpoint"; % save net in folder after each epoch

% pre-allocation for scores
scoresMat                       = zeros(3, 8, generalInfo.numIters);
scoresPerAgeGenderMat           = zeros(4, 7, generalInfo.numIters);
scoresPerDatasetMat             = zeros(3, 7, generalInfo.numIters);
scoresPerSymptoms2classesMat    = zeros(2, 7, generalInfo.numIters);
scoresPerSymptoms4classesMat    = zeros(4, 5, generalInfo.numIters);
lblsPredTrue = cell(generalInfo.numIters, 3);
tic;

% loop over all iterations (classification per iteration):

for iter = 1 : generalInfo.numIters
    
    % Streamline audio feature extraction (mel-spectrogram)
    [afe, winInfo] = setAudioFeatureExtractor(winInfo);
    
    % load data, shuffle and split to train val test
    [adsTrain, adsVal, adsTest, adsSpecsTable] = ...
        loadSplitShuffleDatasets(generalInfo.datasetNames, generalInfo.splitPrcnts, generalInfo.shuffleDatasets, generalInfo.changeSubjectsOrder);
    
    % preprocess datasets (extract auto-segmentation results from yamnet model)
    addAugment = "no";
    [FeaturesTrain, trueLabelsPerAudioTrain, adsSpecsPerAudioTrain] = ...
        preprocessClassifyCrnn(adsTrain, winInfo, afe, addAugment, generalInfo.coughEventDetect);
    [FeaturesVal, trueLabelsPerAudioVal, adsSpecsPerAudioVal] = ...
        preprocessClassifyCrnn(adsVal, winInfo, afe, addAugment, generalInfo.coughEventDetect);
    [FeaturesTest, trueLabelsPerAudioTest, adsSpecsPerAudioTest] = ...
        preprocessClassifyCrnn(adsTest, winInfo, afe, addAugment, generalInfo.coughEventDetect);
    
    % set classes and classes weights
    uniqueLabels = unique(trueLabelsPerAudioVal); % ["negative" "positive"]
    numClasses = length(uniqueLabels); % No. classes
    numTrainPos = sum(trueLabelsPerAudioTrain == "positive");
    numTrainNeg = sum(trueLabelsPerAudioTrain == "negative");
    numTrain = numTrainPos + numTrainNeg;
    
    % give weight to each class based on its opposite pravelance
    classWeights = numTrain ./ ([numTrainNeg numTrainPos] * numClasses);
    
    %     % no weights (values are: [1 1])
    %     classWeights = "none";
    
    % create LSTM network
    lstmLayers = [
        bilstmLayer(hyperParameters.numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm')
        dropoutLayer(hyperParameters.dropProb, 'Name', 'drop')
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
    lgraph = addLayers(lgraph, lgraphCnn.Layers);
    lgraph = connectLayers(lgraph, "fold/out", string(lgraphCnn.Layers(1, 1).Name));
    
    % add unfolding and lstm layers
    layers = [
        sequenceUnfoldingLayer('Name', 'unfold') % change structure back for lstm
        flattenLayer('Name', 'flatten') % change dim from [1, 1, No. output features] to [No. output features, 1]
        lstmLayers
        ];
    
    % connect unfolding and lstm layers to output of CNN
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, string(lgraphCnn.Layers(end, 1).Name), "unfold/in");
    
    % add parallel connection between folding & unfolding layers (needed to restore the sequence structure)
    lgraph = connectLayers(lgraph, "fold/miniBatchSize", "unfold/miniBatchSize");
    
    % specify training options
    validationFrequency = floor(numel(trueLabelsPerAudioTrain)/hyperParameters.miniBatchSize);
    options = trainingOptions( ...
        'adam', ...
        'Plots', 'none', ... % 'none' (default) | 'training-progress'
        'Verbose', false, ... % 1 (true) (default) | 0 (false)
        'MaxEpochs', hyperParameters.MaxEpochs, ...
        'MiniBatchSize', hyperParameters.miniBatchSize, ...
        'Shuffle', 'every-epoch', ... % 'once' (default) | 'never' | 'every-epoch'
        ...
        'ValidationData', {FeaturesVal, trueLabelsPerAudioVal}, ...
        'ValidationFrequency', validationFrequency, ...
        'InitialLearnRate', hyperParameters.InitialLearnRate, ...
        'LearnRateSchedule', hyperParameters.LearnRateSchedule, ... % 'none' (default) | 'piecewise'
        'LearnRateDropPeriod', hyperParameters.LearnRateDropPeriod, ...
        'LearnRateDropFactor', hyperParameters.LearnRateDropFactor, ...
        'L2Regularization', hyperParameters.L2Regularization, ...
        'ValidationPatience', hyperParameters.ValidationPatience, ...
        ...
        'SequencePaddingDirection', hyperParameters.SequencePaddingDirection, ...
        'SequenceLength', hyperParameters.SequenceLength, ...% "longest" (default) | "shortest" | positive integer
        ...
        'CheckpointPath', hyperParameters.checkpointPath, ...
        'BatchNormalizationStatistics', 'moving' ... % 'population' (default) | 'moving'
        );
    
    % remove saved nets from previous training
    delete(hyperParameters.checkpointPath + "/*");
    
    % train network
    [~, net_info] = trainNetwork(FeaturesTrain, trueLabelsPerAudioTrain, lgraph, options);
    
    % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
    net_classification_CRNN = extractNetFromCheckpoint(net_info, hyperParameters.checkpointPath);
    
    % remove saved nets after training
    delete(hyperParameters.checkpointPath + "/*");
    
    % predict on train, val and test sets (probability [0,1])
    miniBatchSize_predict = 1;
    positive_indx = find(string(uniqueLabels) == "positive");
    predLabelsPerAudioTrain = ...
        predict(net_classification_CRNN, FeaturesTrain, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsPerAudioTrain = predLabelsPerAudioTrain(:, positive_indx);
    predLabelsPerAudioVal   = ...
        predict(net_classification_CRNN, FeaturesVal, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsPerAudioVal   = predLabelsPerAudioVal(:, positive_indx);
    predLabelsPerAudioTest  = ...
        predict(net_classification_CRNN, FeaturesTest, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsPerAudioTest  = predLabelsPerAudioTest(:, positive_indx);
    
    % measure time to predict on test set:
    % featureExtraction + predictPerAudio + preAudio2perSubject
    
    % measure testing time
    timeTest(iter) = toc;
    
    [FeaturesTest, trueLabelsPerAudioTest, adsSpecsPerAudioTest] = ...
        preprocessClassifyCrnn(adsTest, winInfo, afe, addAugment, generalInfo.coughEventDetect);
    predLabelsPerAudioTest  = ...
        predict(net_classification_CRNN, FeaturesTest, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsPerAudioTest  = predLabelsPerAudioTest(:, positive_indx);
    [predLabelsPerSubjTest, trueLabelsPerSubjTest, adsSpecsPerSubjTest] = ...
        perAudio2perSubject(predLabelsPerAudioTest, trueLabelsPerAudioTest, adsSpecsPerAudioTest);
    
    % measure testing time
    timeTest(iter) = toc - timeTest(iter);
    
    % change predictions and true labels from per audio signal to per subject
    [predLabelsPerSubjTrain, trueLabelsPerSubjTrain, adsSpecsPerSubjTrain] = ...
        perAudio2perSubject(predLabelsPerAudioTrain, trueLabelsPerAudioTrain, adsSpecsPerAudioTrain);
    [predLabelsPerSubjVal, trueLabelsPerSubjVal, adsSpecsPerSubjVal] = ...
        perAudio2perSubject(predLabelsPerAudioVal, trueLabelsPerAudioVal, adsSpecsPerAudioVal);
    [predLabelsPerSubjTest, trueLabelsPerSubjTest, adsSpecsPerSubjTest] = ...
        perAudio2perSubject(predLabelsPerAudioTest, trueLabelsPerAudioTest, adsSpecsPerAudioTest);
    
    % gather results in cells
    lblsPredCell = {predLabelsPerSubjTrain predLabelsPerSubjVal predLabelsPerSubjTest};
    lblsTrueCell = {trueLabelsPerSubjTrain trueLabelsPerSubjVal trueLabelsPerSubjTest};
    adsSpecsPerSubjCell = {adsSpecsPerSubjTrain adsSpecsPerSubjVal adsSpecsPerSubjTest};
    
    % save predictions & true lables
    lblsPredTrue(iter, :) = {lblsPredCell, lblsTrueCell adsSpecsPerSubjCell};
    
    % find best ROC threshold for val set
    setRocThreshold = "auto"; % "auto" | "manual"
    plotRoc = "no";
    rocThreshold = findBestThresholdAndPlotROC(lblsPredCell{2}, lblsTrueCell{2}, ...
        setRocThreshold, plotRoc, hyperParameters.specificityThershold);
    
    % loop over train, val and test sets. plot scores on all and add confusion
    % matrix on test set only
    for i = 1 : length(lblsPredCell)
        
        % calculate scores
        plotConfusionMat = "no";
        scoresStruct = calculateScores(lblsPredCell{i}, lblsTrueCell{i}, rocThreshold, plotConfusionMat);
        
        % put values in matrix
        scoresMat(i, 1 : end - 1, iter) = cell2mat(struct2cell(scoresStruct))';
    end
    
    % calculate train and val losses
    [val_loss, val_loss_indx] = min(net_info.ValidationLoss);
    train_loss = ...
        mean(net_info.TrainingLoss(max(val_loss_indx - validationFrequency + 1, 1) : val_loss_indx));
    
    % add losses to scoresMat
    scoresMat(:, end, iter) = [train_loss ; val_loss ; nan];
    
    % calculate scores per age / gender for test set
    scoresPerAgeGenderStruct = ...
        calculateScoresPerAgeGender(lblsPredCell{3}, lblsTrueCell{3}, rocThreshold, adsSpecsPerSubjCell{3});
    scoresPerAgeGenderMat(:, :, iter) = cell2mat(struct2cell(scoresPerAgeGenderStruct))';
    
    % calculate scores per dataset for test set
    scoresPerDatasetMat(:, :, iter) = ...
        cell2mat(struct2cell(calculateScoresPerDataset(...
        lblsPredCell{3}, lblsTrueCell{3}, rocThreshold, adsSpecsPerSubjCell{3})))';
    
    % calculate scores per symptoms for test set
    scoresPerSymptoms2classesMat(:, :, iter) = ...
        cell2mat(struct2cell(calculateScoresPerSymptoms2classes(...
        lblsPredCell{3}, lblsTrueCell{3}, rocThreshold, adsSpecsPerSubjCell{3})))';
    scoresPerSymptoms4classesTable = ...
        calculateScoresPerSymptoms4classes(lblsPredCell{3}, lblsTrueCell{3}, rocThreshold, adsSpecsPerSubjCell{3});
    scoresPerSymptoms4classesMat(:, :, iter) = table2array(scoresPerSymptoms4classesTable);
    
    % average scores
    scoresAvg = ...
        string(round(mean(scoresMat(:, :, 1 : iter), 3), 1)) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scoresMat(:, :, 1 : iter), 0, 3), 1));
    
    % average scores per age / gender
    scoresPerAgeGenderAvg = ...
        string(round(mean(scoresPerAgeGenderMat(:, :, 1 : iter), 3), 1)) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scoresPerAgeGenderMat(:, :, 1 : iter), 0, 3), 1));
    
    % average scores per dataset
    scoresPerDatasetAvg = ...
        string(round(mean(scoresPerDatasetMat(:, :, 1 : iter), 3, 'omitnan'), 1)) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scoresPerDatasetMat(:, :, 1 : iter), 0, 3, 'omitnan'), 1));
    
    % average scores per symptoms 2 classes
    scoresPerSymptoms2classesAvg = ...
        string(round(mean(scoresPerSymptoms2classesMat(:, :, 1 : iter), 3, 'omitnan'), 1)) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scoresPerSymptoms2classesMat(:, :, 1 : iter), 0, 3, 'omitnan'), 1));
    
    % average scores per symptoms 4 classes
    scoresPerSymptoms4classesAvg = ...
        string(round(mean(scoresPerSymptoms4classesMat(:, :, 1 : iter), 3, 'omitnan'), 1)) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scoresPerSymptoms4classesMat(:, :, 1 : iter), 0, 3, 'omitnan'), 1));
    
    % remove previous text from command window
    clc;
    
    % display scores
    scoresNames = ...
        ["" "accuracy" "uar" "F1" "sensitivity" "PPV" "specificity" "auc" "loss"];
    scoresPerSymptomsNames = ...
        ["" "No. correct" "overAll" "Accuracy [%]" "Recall (Sensitivity) [%]" "Precision (PPV) [%]"];
    disp('scores:');
    disp(scoresNames(2 : end));
    disp(scoresMat(:, :, 1 : iter));
    disp('No. subjects/records');
    disp(adsSpecsTable);
    disp('average scores:');
    disp([scoresNames ; [["train:" ; "val:" ; "test:"] scoresAvg]]);
    disp('average scores per age / gender:');
    disp([scoresNames(1 : end - 1) ; [["AgeLow:" ; "AgeHigh:" ; "male:" ; "female:"] scoresPerAgeGenderAvg]]);
    disp('average scores per dataset:');
    disp([scoresNames(1 : end - 1) ; [["Coswara ds:" ; "COUGHVID ds:" ; "Voca ds:"] scoresPerDatasetAvg]]);
    disp('average scores per symptoms:');
    disp([scoresNames(1 : end - 1) ; [["symptomsYes:" ; "symptomsNo:"] scoresPerSymptoms2classesAvg]]);
    disp([scoresPerSymptomsNames ; [["posSymptomsYes:" ; "posSymptomsNo:" ; "negSymptomsYes:" ; "negSymptomsNo:"] scoresPerSymptoms4classesAvg]]);
    
    % display progress done and time
    generalInfo.timeElapsed = dispProgressAndTime(generalInfo.numIters, iter);
end

% deselects the GPU device and clears its memory
gpuDevice([]);

%% save results & predictions in mat and csv files:

% % change all results to cells format:
% 
% generalInfoCell = [fieldnames(generalInfo)' ; struct2cell(generalInfo)'];
% generalInfoCell{2,3} = strjoin(generalInfo.datasetNames);
% generalInfoCell{2,4} = strjoin(string(generalInfo.splitPrcnts));
% 
% winInfoCell = [fieldnames(winInfo)' ; struct2cell(winInfo)'];
% winInfoCell(:,3) = []; % remove win vector
% 
% hyperParametersCell = [fieldnames(hyperParameters)' ; struct2cell(hyperParameters)'];
% 
% scoresAvgCell = cellstr([scoresNames ; [["train:" ; "val:" ; "test:"] scoresAvg]]);
% 
% scoresPerAgeGenderAvgCell = ...
%     cellstr([scoresNames(1 : end - 1) ; [["AgeLow:" ; "AgeHigh:" ; "male:" ; "female:"] scoresPerAgeGenderAvg]]);
% 
% scoresPerDatasetAvgCell = ...
%     cellstr([scoresNames(1 : end - 1) ; [["Coswara ds:" ; "COUGHVID ds:" ; "Voca ds:"] scoresPerDatasetAvg]]);
% 
% scoresPerSymptoms2classesAvgCell = ...
%     cellstr([scoresNames(1 : end - 1) ; [["symptomsYes:" ; "symptomsNo:"] scoresPerSymptoms2classesAvg]]);
% 
% scoresPerSymptoms4classesAvgCell = ...
%     cellstr([scoresPerSymptomsNames ; [["posSymptomsYes" ; "posSymptomsNo" ; "negSymptomsYes" ; "negSymptomsNo"] scoresPerSymptoms4classesAvg]]);
% 
% % path and files names
% filePath = "results/";
% fileName = "classifiyCrnnResults " + ...
%     string(datetime, 'yyyy-MM-dd HH-mm') + ...
%     " numIters=" + string(generalInfo.numIters) + ...
%     " coughEventDetect=" + generalInfo.coughEventDetect + ...
%     " RandomSeed=" + string(generalInfo.RandomSeed);
% 
% % save in mat file
% save(filePath + "mat files/" + fileName + ".mat", ...
%     'lblsPredTrue', 'scoresMat', ...
%     'scoresPerAgeGenderMat', 'scoresPerDatasetMat', ...
%     'scoresPerSymptoms2classesMat', 'scoresPerSymptoms4classesMat', ...
%     'generalInfo', 'winInfo', 'hyperParameters', ...
%     'scoresAvg', ...
%     'scoresPerAgeGenderAvg', 'scoresPerDatasetAvg', ...
%     'scoresPerSymptoms2classesAvg', 'scoresPerSymptoms4classesAvg');
% 
% % save in excel file
% writecell(generalInfoCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'overwrite');
% writecell(winInfoCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
% writecell(hyperParametersCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
% writecell(scoresAvgCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
% writecell(scoresPerAgeGenderAvgCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
% writecell(scoresPerDatasetAvgCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
% writecell(scoresPerSymptoms2classesAvgCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
% writecell(scoresPerSymptoms4classesAvgCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
