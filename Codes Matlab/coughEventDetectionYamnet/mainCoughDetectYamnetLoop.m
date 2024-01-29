%% Cough Event Detection - transfer learning on Yamnet model

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpuSelected = "0"; % "0" | "1" | "no"
selectGpu(gpuSelected);

%% run CNN in loop

% iterations options
generalInfo.numIters = 100;
% generalInfo.rocThreshold = 0.6;
% generalInfo.specificityThershold = 0.65;
generalInfo.remLowMagSeg = "yes"; % "yes" | "no"

% save results?
generalInfo.saveResultsInFiles = "yes"; % "yes" | "no"

% datasets loading and spliting info
generalInfo.datasetNames = ["voca" "coughvid" "coswara"]; % ["voca" "coughvid" "coswara"]
generalInfo.splitPrcnts = [0.8 0.1 0.1]; % size of train val test in [%]
generalInfo.shuffleDatasets = "trainVal"; % "all" | "trainVal" | "no"
generalInfo.RandomSeed = 1; % 1 | "shuffle"
rng(generalInfo.RandomSeed);

% add yamnet to serach path
downloadFolder = fullfile(tempdir,'YAMNetDownload');
loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/yamnet.zip');
YAMNetLocation = tempdir;
unzip(loc,YAMNetLocation);
addpath(fullfile(YAMNetLocation,'yamnet'));
netYamnet = yamnet;

% Create audioFeatureExtractor object from audio signals
% best choices:
% window length = 128 [samples]
% overlap coefficient between windows = 50 [%]
% overlap coefficient between segments = 90 [%]

% win & spectrogram length/overlap info
winInfo.winLen = 128;
winInfo.overlapCoeffWin = 0.5;
winInfo.overlapCoeffSpectrograms = 0.9;

% win & spectrogram normalization info
winInfo.SpectrumType = "magnitude"; % "power" "magnitude"
winInfo.WindowNormalization = true; % true false
winInfo.FilterBankNormalization = "bandwidth"; % "area" "bandwidth" "none"

% training options
hyperParameters.MaxEpochs = 10;
hyperParameters.miniBatchSize = 32;
hyperParameters.InitialLearnRate = 0.00003;
hyperParameters.L2Regularization = 0.003;
hyperParameters.LearnRateSchedule = "piecewise"; % "none" "piecewise"
hyperParameters.LearnRateDropPeriod = 1; % 1 2
hyperParameters.LearnRateDropFactor = 0.1;
hyperParameters.ValidationPatience = 4; % positive integer | Inf
hyperParameters.checkpointPath = "checkpoint"; % save net in folder after each epoch

% pre-allocation for scores
scoresMat = zeros(3, 7, generalInfo.numIters);
lblsPredTrue = cell(generalInfo.numIters, 3);
tic;

% loop over all iterations (classification per iteration):
for iter = 1 : generalInfo.numIters
    
    % Streamline audio feature extraction (mel-spectrogram)
    [afe, winInfo] = setAudioFeatureExtractor(winInfo);
    
    % load data, shuffle and split to train val test
    [adsTrain, adsVal, adsTest, adsSpecsTable] = loadSplitShuffleDatasets(...
        generalInfo.datasetNames, ...
        generalInfo.splitPrcnts, ...
        generalInfo.shuffleDatasets);
    
    % Extract features from train and validation sets
    addTrueLbls = "yes";
    remLowMagSeg = "yes";
    [trainFeatures, trainLabels, trainSpecs] = preprocessSegYamnet(adsTrain, afe, winInfo, addTrueLbls, generalInfo.remLowMagSeg);
    [valFeatures, valLabels, valSpecs] = preprocessSegYamnet(adsVal, afe, winInfo, addTrueLbls, generalInfo.remLowMagSeg);
    
    disp(['No. segments train: ', num2str(numel(trainLabels))]);
    disp(['No. segments val: ', num2str(numel(valLabels))]);
    
    % cough segmentation requires only 2 classes (cough / non-cough).
    % Read in YAMNet, convert it to a layerGraph, and then replace the final fullyConnectedLayer and the final classificationLayer to reflect the new task.
    uniqueLabels = unique(trainLabels);
    numClasses = 2;
    numTrainPos = sum(trainLabels == "Cough");
    numTrainNeg = sum(trainLabels == "nonCough");
    numTrain = numTrainPos + numTrainNeg;
    classWeightsUniform = 'none';
    %     classWeightsNonUniform = [numTrainPos, numTrainNeg] / num_train;
    classWeightsNonUniform = numTrain ./ ([numTrainNeg numTrainPos] * numClasses);
    
    classWeights = classWeightsNonUniform;
    
    disp(['numTrainNeg: ', num2str(numTrainNeg)]);
    disp(['numTrainPos: ', num2str(numTrainPos)]);
    disp(['classWeights: ', num2str(classWeights)]);
    
    lgraph = layerGraph(netYamnet.Layers);
    
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
    
    % [net_seg_cough_yamnet, netInfo] = trainNetwork(trainFeatures, trainLabels, lgraph, options);
    
    % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
    netCoughDetectYamnet = extractNetFromCheckpoint(netInfo, hyperParameters.checkpointPath);
    
    % remove saved nets from previous training
    delete(hyperParameters.checkpointPath + "/*");
    
    % predict on test data (probability for each segment for being part of a cough event)
    addTrueLbls = "yes";
    [lblsPredTrain, lblsTrueTrain, adsSpecsTrain] = extractPredictions(netCoughDetectYamnet, adsTest, winInfo, addTrueLbls);
    [lblsPredVal, lblsTrueVal, adsSpecsVal] = extractPredictions(netCoughDetectYamnet, adsTest, winInfo, addTrueLbls);
    [lblsPredTest, lblsTrueTest, adsSpecsTest] = extractPredictions(netCoughDetectYamnet, adsTest, winInfo, addTrueLbls);
    
    % gather results in cells
    lblsPredCell = {lblsPredTrain lblsPredVal lblsPredTest};
    lblsTrueCell = {lblsTrueTrain lblsTrueVal lblsTrueTest};
    adsSpecsPerSubjCell = {adsSpecsTrain adsSpecsVal adsSpecsTest};
    
    % save predictions & true lables
    lblsPredTrue(iter, :) = {lblsPredCell lblsTrueCell adsSpecsPerSubjCell};
    
    % find best ROC threshold based on max F1 on Val set
    plotRoc = "no";
    rocThreshold = findBestThresholdAndPlotROC(lblsPredVal, lblsTrueVal, plotRoc);
    
    % evaluate the performance of the network on the test set:
    plotConfMat = "no";
    scoresCoughDetectYamnet = ...
        extractScoresCoughDetectYamnet(lblsPredTest, lblsTrueTest, rocThreshold, plotConfMat);
    scoresMat(:, :, iter) = table2array(scoresCoughDetectYamnet);
    
    % average scores
    scoresAvg = ...
        string(round(mean(scoresMat(:, :, 1 : iter), 3), 1)) + ...
        ' ' + char(177) + ' ' + ...
        string(round(std(scoresMat(:, :, 1 : iter), 0, 3), 1));
    
    % remove previous text from command window
    clc;
    
    % display scores
    scoresNames = ["per" "accuracy" "uar" "F1" "sensitivity" "PPV" "specificity" "auc"];
    disp('scores:');
    disp(scoresNames(2 : end));
    disp(scoresMat(:, :, 1 : iter));
    disp('average scores:');
    disp([scoresNames ; [["segment" ; "event: 50% overlap" ; "event: 70% overlap"] scoresAvg]]);
    
    % display progress done and time
    generalInfo.timeElapsed = dispProgressAndTime(generalInfo.numIters, iter);
end

% deselects the GPU device and clears its memory
gpuDevice([]);

% plot bar
plotBar(scoresMat);

%% save results & predictions in mat and csv files:

% change all results to cells format

generalInfoCell = [fieldnames(generalInfo)' ; struct2cell(generalInfo)'];
generalInfoCell{2,2} = strjoin(string(generalInfo.splitPrcnts));

winInfoCell = [fieldnames(winInfo)' ; struct2cell(winInfo)'];
winInfoCell(:,3) = []; % remove win vector

hyperParametersCell = [fieldnames(hyperParameters)' ; struct2cell(hyperParameters)'];

scoresAvgCell = cellstr([scoresNames ; [["segment" ; "event: 50% overlap" ; "event: 70% overlap"] scoresAvg]]);

% path and files names
filePath = "results/";
fileName = "coughDetectYamnetResults " + string(datetime, 'yyyy-MM-dd HH-mm') + ...
    " numIters=" + string(generalInfo.numIters) + ...
    " RandomSeed=" + string(generalInfo.RandomSeed);

% save results and info in files
switch saveResultsInFiles
    
    case "yes"
        
        % save in mat file
        save(filePath + "mat files/" + fileName + ".mat", ...
            'lblsPredTrue', 'scoresMat', ...
            'generalInfo', 'winInfo', 'hyperParameters', 'scoresAvg', ...
            'netCoughDetectYamnet', 'rocThreshold');
        
        % save in excel file
        writecell(generalInfoCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'overwrite');
        writecell(winInfoCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
        writecell(hyperParametersCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
        writecell(scoresAvgCell, filePath + "excel files/" + fileName + ".xls", 'WriteMode', 'append');
        
        % inform saving
        disp("results and info were saves");
        
    case "no"
        
        % do nothing
        disp("no saving..");
end
