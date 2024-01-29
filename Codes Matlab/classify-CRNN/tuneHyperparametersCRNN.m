%% Hyper-parameters tuning - CRNN - covid-19 classification - cough audio signals

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpuSelected = "1"; % "0" | "1" | "no"
selectGpu(gpuSelected);

%% run RCNN

% iterations options
generalInfo.numIters = 100;
generalInfo.numCV = 5; % No. CV
% generalInfo.rocThreshold = 0.6;
generalInfo.specificityThershold = 0;
generalInfo.remLowMagSeg = "no"; % "yes" | "no"

% save results?
generalInfo.saveResultsInFiles = "yes"; % "yes" | "no"

% datasets loading and spliting info
generalInfo.datasetNames = ["voca" "coughvid" "coswara"]; % ["voca" "coughvid" "coswara"]
generalInfo.splitPrcnts = [0.6 0.2 0.2]; % size of train val test in [%]
generalInfo.shuffleDatasets = "trainVal"; % "all" | "trainVal" | "no"
generalInfo.changeSubjectsOrder = "no"; % "yes" | "no"
generalInfo.RandomSeed = 1; % 1 | "shuffle"
rng(generalInfo.RandomSeed);

% load relevent layers from pre-traind CNN network
netCnnName = "vggish"; % "yamnet" | "vggish" | "openl3"
freezeLayers = "no";
lgraphCnn = loadCnnLgraph(netCnnName, freezeLayers);

% win & spectrogram length/overlap info
initialHyperParameters.winLen = [128 256]; % [128 256]
initialHyperParameters.overlapCoeffWin = [0.5 0.75]; % [0.5 0.75]
initialHyperParameters.overlapCoeffSpectrograms = [0.7 0.75 0.8 0.85]; % [0.75 0.8 0.85 0.9]

% win & spectrogram normalization info
initialHyperParameters.SpectrumType = "magnitude"; % "power" "magnitude"
initialHyperParameters.WindowNormalization = false; % true false
initialHyperParameters.FilterBankNormalization = ["bandwidth" "none"]; % "area" "bandwidth" "none"

% LSTM
initialHyperParameters.numHiddenUnits = [128 256 512 1024]; % [16 32 64]
initialHyperParameters.dropProb = [0.01 0.05 0.1 0.2 0.3]; % 0.05 0.1 0.15 0.2 0.25

initialHyperParameters.MaxEpochs = 30;
initialHyperParameters.miniBatchSize = 4; % [4 8]
initialHyperParameters.InitialLearnRate = [0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1];
initialHyperParameters.L2Regularization = [0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1];
initialHyperParameters.LearnRateSchedule = "piecewise"; % "none" "piecewise"
initialHyperParameters.LearnRateDropPeriod = 1; % 1 2
initialHyperParameters.LearnRateDropFactor = [0.5 0.7 0.9 0.99];
initialHyperParameters.ValidationPatience = 10;
initialHyperParameters.SequencePaddingDirection = "right"; % "right" "left"
hyperParameters.SequenceLength = "longest"; % "longest" (default) | "shortest" | positive integer
hyperParameters.checkpointPath = "checkpoint"; % save net in folder after each epoch

% path and files names for saving data in file
filePath = "tuneHyperparametersResults/";
fileName = "tuneHyperparametersClassifyCrnnResults " + ...
    string(datetime, 'yyyy-MM-dd HH-mm') + ...
    " numIters=" + string(generalInfo.numIters) + ...
    " RandomSeed=" + string(generalInfo.RandomSeed);

% pre-allocation for scores
lblsPredTrue = cell(generalInfo.numIters, 3);
scoresNamesForPlot = ...
    [{"uar"} {"f1"} {"sens"} {"ppv"} {"spec"} {"auc"} ...
    {"lossTrain"} {"lossVal"}];
scoresForPlot = cell(generalInfo.numIters + 1, length(scoresNamesForPlot));
scoresForPlot(1, :) = scoresNamesForPlot;

tic;

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
    
    % lstm - num Hidden Units
    randIndx = randi(length(initialHyperParameters.numHiddenUnits), 1);
    hyperParameters.numHiddenUnits = initialHyperParameters.numHiddenUnits(randIndx);
    
    % lstm - dropout probability
    randIndx = randi(length(initialHyperParameters.dropProb), 1);
    hyperParameters.dropProb = initialHyperParameters.dropProb(randIndx);
    
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
    
    % Sequence Padding Direction
    randIndx = randi(length(initialHyperParameters.SequencePaddingDirection), 1);
    hyperParameters.SequencePaddingDirection = initialHyperParameters.SequencePaddingDirection(randIndx);
    
    % Streamline audio feature extraction (mel-spectrogram)
    [afe, winInfo] = setAudioFeatureExtractor(winInfo);
    
    % pre-allocation for val results from CV
    acc         = zeros(1, generalInfo.numCV);
    uar         = zeros(1, generalInfo.numCV);
    f1          = zeros(1, generalInfo.numCV);
    sensitivity = zeros(1, generalInfo.numCV);
    ppv         = zeros(1, generalInfo.numCV);
    specificity = zeros(1, generalInfo.numCV);
    auc         = zeros(1, generalInfo.numCV);
    lossVal     = zeros(1, generalInfo.numCV);
    lossTrain   = zeros(1, generalInfo.numCV);
    
    % loop k-fold times for CV
    for i = 1 : generalInfo.numCV
        
        % load data, shuffle and split to train val test
        [adsTrain, adsVal, adsTest, adsSpecsTable] = loadSplitShuffleDatasets(...
            generalInfo.datasetNames, ...
            generalInfo.splitPrcnts, ...
            generalInfo.shuffleDatasets, ...
            generalInfo.changeSubjectsOrder);
        
        % preprocess datasets (extract auto-segmentation results from yamnet model)
        coughEventDetect = "yes";
        addAugment = "no";
        [FeaturesTrain, trueLabelsPerAudioTrain, adsSpecsTrain] = ...
            preprocessClassifyCrnn(adsTrain, winInfo, afe, addAugment, coughEventDetect);
        addAugment = "no";
        [FeaturesVal, trueLabelsPerAudioVal, adsSpecsVal] = ...
            preprocessClassifyCrnn(adsVal, winInfo, afe, addAugment, coughEventDetect);
        
        % set classes and classes weights
        uniqueLabels = unique(trueLabelsPerAudioTrain); % ["negative" "positive"]
        numClasses = length(uniqueLabels); % No. classes
        numTrainPos = sum(trueLabelsPerAudioTrain == "positive");
        numTrainNeg = sum(trueLabelsPerAudioTrain == "negative");
        numTrain = numTrainPos + numTrainNeg;
        
        % give weight to each class based on its opposite pravelance
        classWeights = numTrain ./ ([numTrainNeg numTrainPos] * numClasses);
        
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
        [~, netInfo] = trainNetwork(FeaturesTrain, trueLabelsPerAudioTrain, lgraph, options);
        
        % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
        netClassifyCrnn = extractNetFromCheckpoint(netInfo, hyperParameters.checkpointPath);
        
        % remove saved nets after training
        delete(hyperParameters.checkpointPath + "/*");
        
        % predict on validation set
        miniBatchSizePredict = 1;
        positiveIndx = find(string(uniqueLabels) == "positive");
        predLabelsPerAudioVal   = ...
            predict(netClassifyCrnn, FeaturesVal, 'MiniBatchSize', miniBatchSizePredict);
        predLabelsPerAudioVal = predLabelsPerAudioVal(:, positiveIndx);
        
        % change predictions and true labels from per audio signal to per subject
        [predLabelsPerSubjVal, trueLabelsPerSubjVal] = ...
            perAudio2perSubject(predLabelsPerAudioVal, trueLabelsPerAudioVal, adsSpecsVal);
        
        % find best ROC threshold for val set
        setRocThreshold = "auto";
        plotRoc = "no";
        rocThreshold = ...
            findBestThresholdAndPlotROC(predLabelsPerSubjVal, trueLabelsPerSubjVal, ...
            setRocThreshold, plotRoc, generalInfo.specificityThershold);
        
        % calculate scores
        plotConfusionMat = "no";
        scoresStruct = calculateScores(...
            predLabelsPerSubjVal, trueLabelsPerSubjVal, rocThreshold, plotConfusionMat);
        acc(i)          = scoresStruct.acc;
        uar(i)          = scoresStruct.uar;
        f1(i)           = scoresStruct.F1;
        sensitivity(i)  = scoresStruct.sensitivity;
        ppv(i)          = scoresStruct.PPV;
        specificity(i)  = scoresStruct.specificity;
        auc(i)          = scoresStruct.auc;
        
        % loss
        [lossVal(i), lossValIndx] = min(netInfo.ValidationLoss);
        lossTrain(i) = ...
            mean(netInfo.TrainingLoss(max(lossValIndx - validationFrequency + 1, 1) : lossValIndx));
    end
    
    scores.acc          = mean(acc);
    scores.uar          = mean(uar);
    scores.f1           = mean(f1);
    scores.sensitivity  = mean(sensitivity);
    scores.ppv          = mean(ppv);
    scores.specificity  = mean(specificity);
    scores.auc          = mean(auc);
    scores.lossVal      = mean(lossVal);
    scores.lossTrain    = mean(lossTrain);
    
    % save results & predictions in csv file:
    % change all results to cells format:
    
    generalInfoCell = [fieldnames(generalInfo)' ; struct2cell(generalInfo)'];
    generalInfoCell{2,6} = strjoin(generalInfo.datasetNames);
    generalInfoCell{2,7} = strjoin(string(generalInfo.splitPrcnts));
    
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
        {scores.uar} {scores.f1} {scores.sensitivity} {scores.ppv} {scores.specificity} {scores.auc} ...
        {scores.lossTrain} {scores.lossVal} ...
        ];
    disp(scoresForPlot(1 : iter + 1, :));
    
    % display progress done and time
    generalInfo.timeElapsed = dispProgressAndTime(generalInfo.numIters, iter);
end

% deselects the GPU device and clears its memory
gpuDevice([]);
