%% Hyper-parameters tuning - yamnet - cough segmentation - cough audio signals

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpuSelected = "1"; % "0" | "1" | "no"
selectGpu(gpuSelected);

%% run model

% iterations options
generalInfo.numIters = 100;
generalInfo.numCV = 5; % No. CV
% generalInfo.rocThreshold = 0.6;
% generalInfo.specificityThershold = 0.65;
generalInfo.remLowMagSeg = "yes"; % "yes" | "no"

% save results?
generalInfo.saveResultsInFiles = "yes"; % "yes" | "no"

% datasets loading and spliting info
generalInfo.datasetNames = ["voca" "coughvid" "coswara"]; % ["voca" "coughvid" "coswara"]
generalInfo.splitPrcnts = [0.8 0.1 0.1]; % size of train val test in [%]
generalInfo.shuffleDatasets = "trainVal"; % "all" | "trainVal" | "no"
generalInfo.RandomSeed = "shuffle"; % 1 | "shuffle"
rng(generalInfo.RandomSeed);

% load CNN - YAMNet
downloadFolder = fullfile(tempdir,'YAMNetDownload');
loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/yamnet.zip');
YAMNetLocation = tempdir;
unzip(loc,YAMNetLocation);
addpath(fullfile(YAMNetLocation,'yamnet'));
net = yamnet;

% win & spectrogram length/overlap info
initialHyperParameters.winLen = [128 256]; % [128 256]
initialHyperParameters.overlapCoeffWin = [0.5 0.75]; % [0.5 0.75]
initialHyperParameters.overlapCoeffSpectrograms = 0.9; % [0.75 0.8 0.85 0.9]

% win & spectrogram normalization info
initialHyperParameters.SpectrumType = "magnitude"; % ["power" "magnitude"]
initialHyperParameters.WindowNormalization = false; % [true false]
initialHyperParameters.FilterBankNormalization = ["bandwidth" "none"]; % "area" "bandwidth" "none"

% training options
initialHyperParameters.MaxEpochs = 15; % [10 15 20]
initialHyperParameters.miniBatchSize = [32 64 128 256];
initialHyperParameters.InitialLearnRate = [0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01];
initialHyperParameters.L2Regularization = [0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01];
initialHyperParameters.LearnRateSchedule = "piecewise"; % "none" "piecewise"
initialHyperParameters.LearnRateDropPeriod = 1; % 1 2
initialHyperParameters.LearnRateDropFactor = [0.1 0.5 0.75 0.9 0.99];
initialHyperParameters.ValidationPatience = 6; % positive integer | Inf | [4 6 8 10]
hyperParameters.checkpointPath = "checkpoint"; % save net in folder after each epoch

% path and files names for saving data in file
filePath = "tuneHyperparametersResults/";
fileName = "tuneHyperparametersCoughDetectResults " + ...
    string(datetime, 'yyyy-MM-dd HH-mm') + ...
    " numIters=" + string(generalInfo.numIters) + ...
    " RandomSeed=" + string(generalInfo.RandomSeed);

% pre-allocation for scores
lblsPredTrue = cell(generalInfo.numIters, 3);
scoresNamesForPlot = ...
    [{"uarSeg"} {"aucSeg"} {"f1Event70"} {"sensEvent70"} {"ppvEvent70"} ...
    {"lossTrain"} {"lossVal"}];
scoresForPlot = cell(generalInfo.numIters + 1, length(scoresNamesForPlot));
scoresForPlot(1, :) = scoresNamesForPlot;

tic;

% loop over all iterations (desicion per iteration):

for iter = 1 : generalInfo.numIters
    
    % win length & and overlap coeff
    randIndx = randi(length(initialHyperParameters.winLen), 1);
    winInfo.winLen = initialHyperParameters.winLen(randIndx);
    winInfo.overlapCoeffWin = initialHyperParameters.overlapCoeffWin(randIndx);
    
    % overlap between spectrograms
    randIndx = randi(length(initialHyperParameters.overlapCoeffSpectrograms), 1);
    winInfo.overlapCoeffSpectrograms = initialHyperParameters.overlapCoeffSpectrograms(randIndx);
    
    % spectrum type
    randIndx = randi(length(initialHyperParameters.SpectrumType), 1);
    winInfo.SpectrumType = initialHyperParameters.SpectrumType(randIndx);
    
    % win norm
    randIndx = randi(length(initialHyperParameters.WindowNormalization), 1);
    winInfo.WindowNormalization = initialHyperParameters.WindowNormalization(randIndx);
    
    % filter bank norm
    randIndx = randi(length(initialHyperParameters.FilterBankNormalization), 1);
    winInfo.FilterBankNormalization = initialHyperParameters.FilterBankNormalization(randIndx);
    
    % maximum No. epochs
    randIndx = randi(length(initialHyperParameters.MaxEpochs), 1);
    hyperParameters.MaxEpochs = initialHyperParameters.MaxEpochs(randIndx);
    
    % mini batch size
    randIndx = randi(length(initialHyperParameters.miniBatchSize), 1);
    hyperParameters.miniBatchSize = initialHyperParameters.miniBatchSize(randIndx);
    
    % InitialLearn Rate
    randIndx = randi(length(initialHyperParameters.InitialLearnRate), 1);
    hyperParameters.InitialLearnRate = initialHyperParameters.InitialLearnRate(randIndx);
    
    % L2 Regularization
    randIndx = randi(length(initialHyperParameters.L2Regularization), 1);
    hyperParameters.L2Regularization = initialHyperParameters.L2Regularization(randIndx);
    
    % Learn Rate Schedule
    randIndx = randi(length(initialHyperParameters.LearnRateSchedule), 1);
    hyperParameters.LearnRateSchedule = initialHyperParameters.LearnRateSchedule(randIndx);
    
    % Learn Rate Drop Period
    randIndx = randi(length(initialHyperParameters.LearnRateDropPeriod), 1);
    hyperParameters.LearnRateDropPeriod = initialHyperParameters.LearnRateDropPeriod(randIndx);
    
    % Learn Rate Drop Factor
    randIndx = randi(length(initialHyperParameters.LearnRateDropFactor), 1);
    hyperParameters.LearnRateDropFactor = initialHyperParameters.LearnRateDropFactor(randIndx);
    
    % Validation Patience
    randIndx = randi(length(initialHyperParameters.ValidationPatience), 1);
    hyperParameters.ValidationPatience = initialHyperParameters.ValidationPatience(randIndx);
    
    % Streamline audio feature extraction (mel-spectrogram)
    [afe, winInfo] = setAudioFeatureExtractor(winInfo);
    
    % pre-allocation for val results from CV:
    
    % per segment
    accValSeg          = zeros(1, generalInfo.numCV);
    uarValSeg          = zeros(1, generalInfo.numCV);
    f1ValSeg           = zeros(1, generalInfo.numCV);
    sensitivityValSeg  = zeros(1, generalInfo.numCV);
    ppvValSeg          = zeros(1, generalInfo.numCV);
    specificityValSeg  = zeros(1, generalInfo.numCV);
    aucValSeg          = zeros(1, generalInfo.numCV);
    
    % per event 50% overlap
    f1ValEvent50           = zeros(1, generalInfo.numCV);
    sensitivityValEvent50  = zeros(1, generalInfo.numCV);
    ppvValEvent50          = zeros(1, generalInfo.numCV);
    
    % per event 70% overlap
    f1ValEvent70           = zeros(1, generalInfo.numCV);
    sensitivityValEvent70  = zeros(1, generalInfo.numCV);
    ppvValEvent70          = zeros(1, generalInfo.numCV);
    
    % loss
    lossVal = zeros(1, generalInfo.numCV);
    lossTrain = zeros(1, generalInfo.numCV);
    
    % loop k-fold times for CV
    for i = 1 : generalInfo.numCV
        
        % load data, shuffle and split to train val test
        [adsTrain, adsVal, adsTest, adsSpecsTable] = loadSplitShuffleDatasets(...
            generalInfo.datasetNames, ...
            generalInfo.splitPrcnts, ...
            generalInfo.shuffleDatasets);
        
        % Extract features from train and validation sets
        addTrueLbls = "yes";
        [trainFeatures, trainLabels, trainSpecs] = preprocessSegYamnet(adsTrain, afe, winInfo, addTrueLbls, generalInfo.remLowMagSeg);
        [valFeatures, valLabels, valSpecs] = preprocessSegYamnet(adsVal, afe, winInfo, addTrueLbls, generalInfo.remLowMagSeg);
        
        % cough segmentation requires only 2 classes (cough / non-cough).
        % replace final fullyConnectedLayer and the final classificationLayer to reflect the new task.
        uniqueLabels = unique(trainLabels);
        numClasses = 2;
        numTrainPos = sum(trainLabels == "Cough");
        numTrainNeg = sum(trainLabels == "nonCough");
        numTrain = numTrainPos + numTrainNeg;
        classWeights = numTrain ./ ([numTrainNeg numTrainPos] * numClasses);
        
        net = yamnet;
        
        lgraph = layerGraph(net.Layers);
        
        newDenseLayer = fullyConnectedLayer(numClasses, "Name", "dense");
        lgraph = replaceLayer(lgraph, "dense", newDenseLayer);
        
        newClassificationLayer = classificationLayer( ...
            "Name", "output", ...
            "Classes", uniqueLabels, ...
            'ClassWeights', classWeights);
        lgraph = replaceLayer(lgraph, "Sound", newClassificationLayer);
        
        % Define training options
        validationFrequency = floor(numel(trainLabels)/hyperParameters.miniBatchSize);
        
        options = trainingOptions( ...
            'adam', ...
            'Plots', 'none', ... % 'none' (default) | 'training-progress'
            'Verbose', false, ... % 1 (true) (default) | 0 (false)
            'MaxEpochs', hyperParameters.MaxEpochs, ...
            'MiniBatchSize', hyperParameters.miniBatchSize, ...
            'Shuffle', 'every-epoch', ... % 'once' (default) | 'never' | 'every-epoch'
            'ExecutionEnvironment', 'auto', ... % 'auto' (default) | 'cpu' | 'gpu' | 'multi-gpu' | 'parallel'
            ...
            'ValidationData', {single(valFeatures), valLabels}, ...
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
        [~, netInfo] = trainNetwork(trainFeatures, trainLabels, lgraph, options);
        
        % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
        netSegCoughYamnet = extractNetFromCheckpoint(netInfo, hyperParameters.checkpointPath);
        
        % remove saved nets from previous training
        delete(hyperParameters.checkpointPath + "/*");
        
        % predict on validation set (per segment)
        addTrueLbls = "yes";
        [lblsPredDecTrain, lblsTrueDecTrain, adsSpecsTrain] = extractPredictions(netSegCoughYamnet, adsTrain, winInfo, addTrueLbls);
        [lblsPredDecVal, lblsTrueDecVal, adsSpecsVal] = extractPredictions(netSegCoughYamnet, adsVal, winInfo, addTrueLbls);
        
%         % gather results in cells
%         lblsPredCell = {lblsPredDecTrain lblsPredDecVal};
%         lblsTrueCell = {lblsTrueDecTrain lblsTrueDecVal};
%         adsSpecsCell = {adsSpecsTrain adsSpecsVal};
%         
%         % save predictions & true lables
%         lblsPredTrue(iter, :) = {lblsPredCell, lblsTrueCell adsSpecsCell};
        
        % find best ROC threshold for val set
        plotRoc = "no";
        rocThreshold = findBestThresholdAndPlotROC(lblsPredDecVal, lblsTrueDecVal, plotRoc);
        
        % calculate scores
        plotConfMat = "no";
        scoresPerCV = ...
            table2array(extractScoresCoughDetectYamnet(lblsPredDecVal, lblsTrueDecVal, rocThreshold, plotConfMat));
        
        % per segment
        accValSeg(i)          = scoresPerCV(1, 1);
        uarValSeg(i)          = scoresPerCV(1, 2);
        f1ValSeg(i)           = scoresPerCV(1, 3);
        sensitivityValSeg(i)  = scoresPerCV(1, 4);
        ppvValSeg(i)          = scoresPerCV(1, 5);
        specificityValSeg(i)  = scoresPerCV(1, 6);
        aucValSeg(i)          = scoresPerCV(1, 7);
        
        % per event 50% overlap
        f1ValEvent50(i)           = scoresPerCV(2, 3);
        sensitivityValEvent50(i)  = scoresPerCV(2, 4);
        ppvValEvent50(i)          = scoresPerCV(2, 5);
        
        % per event 70% overlap
        f1ValEvent70(i)           = scoresPerCV(3, 3);
        sensitivityValEvent70(i)  = scoresPerCV(3, 4);
        ppvValEvent70(i)          = scoresPerCV(3, 5);
        
        % loss
        [lossVal(i), valLossIndx] = min(netInfo.ValidationLoss);
        lossTrain(i) = ...
            mean(netInfo.TrainingLoss(max(valLossIndx - validationFrequency + 1, 1) : valLossIndx));
    end
    
    % take mean value of results:
    
    % per segment
    scores.accValSeg          = mean(accValSeg);
    scores.uarValSeg          = mean(uarValSeg);
    scores.f1ValSeg           = mean(f1ValSeg);
    scores.sensitivityValSeg  = mean(sensitivityValSeg);
    scores.ppvValSeg          = mean(ppvValSeg);
    scores.specificityValSeg  = mean(specificityValSeg);
    scores.aucValSeg          = mean(aucValSeg);
    
    % per event 50% overlap
    scores.f1ValEvent50           = mean(f1ValEvent50);
    scores.sensitivityValEvent50  = mean(sensitivityValEvent50);
    scores.ppvValEvent50          = mean(ppvValEvent50);
    
    % per event 70% overlap
    scores.f1ValEvent70           = mean(f1ValEvent70);
    scores.sensitivityValEvent70  = mean(sensitivityValEvent70);
    scores.ppvValEvent70          = mean(ppvValEvent70);
    
    % loss
    scores.lossVal     = mean(lossVal);
    scores.lossTrain   = mean(lossTrain);
    
    % save results & predictions in csv file:
    % change all results to cells format:
    
    generalInfoCell = [fieldnames(generalInfo)' ; struct2cell(generalInfo)'];
    generalInfoCell{2,5} = strjoin(generalInfo.datasetNames);
    generalInfoCell{2,6} = strjoin(string(generalInfo.splitPrcnts));
    
    winInfoCell = [fieldnames(winInfo)' ; struct2cell(winInfo)'];
    winInfoCell(:,3) = []; % remove win vector
    
    hyperParametersCell = [fieldnames(hyperParameters)' ; struct2cell(hyperParameters)'];
    
    scoresCell = [fieldnames(scores)' ; struct2cell(scores)'];
    
    allInfo = [scoresCell hyperParametersCell winInfoCell generalInfoCell];
    
    % save in excel file
    switch iter
        case 1
            writecell(allInfo, filePath + fileName + ".xls", 'WriteMode', 'append');
        otherwise
            writecell(allInfo(2, :), filePath + fileName + ".xls", 'WriteMode', 'append');
    end
    
    % remove previous text from command window
    clc;
    
    % display scores
    scoresForPlot(iter + 1, :) = [ ...
        {scores.uarValSeg} {scores.aucValSeg} {scores.f1ValEvent70} {scores.sensitivityValEvent70} {scores.ppvValEvent70} ...
        {scores.lossTrain} {scores.lossVal} ...
        ];
    disp(scoresForPlot(1 : iter + 1, :));
    
    % display progress done and time
    generalInfo.timeElapsed = dispProgressAndTime(generalInfo.numIters, iter);
end

% deselects the GPU device and clears its memory
gpuDevice([]);
