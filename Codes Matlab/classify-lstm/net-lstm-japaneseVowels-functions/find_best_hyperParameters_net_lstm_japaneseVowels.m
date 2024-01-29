%%

% remove previous data
close all; clc; clear;

ACC_TEST = [];

for i = 1 : 10
    
    % load dataset (train & test)
    [xTrain, yTrain] = japaneseVowelsTrainData;
    [xTest, yTest] = japaneseVowelsTestData;
    
    % combine train & test to 1 single dataset
    x = [xTrain ; xTest];
    y = [yTrain ; yTest];
    
    % shuffle dataset
    len_ds = length(x);
    indx_rand = randperm(len_ds);
    x = x(indx_rand);
    y = y(indx_rand);
    
    % save dataset x y len_ds
    
    %
    
    % close all; clc; clear;
    % disp("train     val     test");
    % disp("");
    
    %
    
    % % load data
    % load dataset
    
    % split dataset to (train, val, test)
    
    len_train = round(0.6 * len_ds);
    len_val = round(0.2 * len_ds);
    len_test = len_ds - len_val - len_train;
    
    xTrain = x(1 : len_train);
    xVal = x(len_train + 1 : len_train + len_val);
    xTest = x(len_train + len_val + 1 : end);
    
    yTrain = y(1 : len_train);
    yVal = y(len_train + 1 : len_train + len_val);
    yTest = y(len_train + len_val + 1 : end);
    
    inputSize = 12;
    numHiddenUnits = 128;
    numUnitsFc1 = 64;
    numUnitsFc2 = 32;
    numClasses = 9;
    
    layers = [ ...
        sequenceInputLayer(inputSize, 'Name', 'input')
        lstmLayer(numHiddenUnits,'OutputMode','sequence', 'Name', 'lstm1')
        lstmLayer(numHiddenUnits,'OutputMode','sequence', 'Name', 'lstm2')
        lstmLayer(numHiddenUnits,'OutputMode','last', 'Name', 'lstm3')
        fullyConnectedLayer(numUnitsFc1, 'Name', 'fc1')
        batchNormalizationLayer("Name", "bn1")
        reluLayer("Name", "relu1")
        fullyConnectedLayer(numUnitsFc2, 'Name', 'fc2')
        batchNormalizationLayer("Name", "bn2")
        reluLayer("Name", "relu2")
        fullyConnectedLayer(numClasses, "Name", "fc3")
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classification')];
    
    % MaxEpochs = 80;
    % miniBatchSize = 64;
    % InitialLearnRate = 0.001;
    % L2Regularization =0.0001;
    % % validationFrequency = floor(numel(yTrain)/miniBatchSize);
    % validationFrequency = 50;
    %
    % options = trainingOptions( ...
    %     'adam', ...
    %     'MaxEpochs', MaxEpochs, ...
    %     'MiniBatchSize', miniBatchSize, ...
    %     'GradientThreshold', 1, ...
    %     'Verbose',false, ...
    %     'Shuffle', 'every-epoch', ...
    %     'ValidationData', {xVal, yVal}, ...
    %     'ValidationFrequency', validationFrequency, ...
    %     'InitialLearnRate', InitialLearnRate, ...
    %     'LearnRateSchedule', 'none', ...
    %     'LearnRateDropPeriod', 10, ...
    %     'LearnRateDropFactor', 0.1, ...
    %     'L2Regularization', L2Regularization, ...
    %     'SequencePaddingDirection', 'right' ...
    %     );
    
    % maxEpochs = 70;
    % miniBatchSize = 27;
    %
    % options = trainingOptions('adam', ...
    %     'ExecutionEnvironment','auto', ...
    %     'MaxEpochs',maxEpochs, ...
    %     'MiniBatchSize',miniBatchSize, ...
    %     'GradientThreshold',1, ...
    %     'Verbose',false, ...
    %     'ValidationData', {xVal, yVal}, ...
    %     'Shuffle', 'every-epoch' ...
    %     );
    
    % options = trainingOptions('adam', ...
    %     'Verbose', false, ...
    %     'MaxEpochs', 70, ...
    %     'MiniBatchSize', 27, ...
    %     'Shuffle', 'every-epoch', ...
    %     'ValidationData', {xVal, yVal}, ...
    %     'ValidationFrequency', 50, ...
    %     'InitialLearnRate', 0.001, ...
    %     'LearnRateSchedule', 'none', ...
    %     'LearnRateDropPeriod', 10, ...
    %     'LearnRateDropFactor', 0.1, ...
    %     'L2Regularization', 0.0001, ...
    %     'SequencePaddingDirection', 'right' ...
    %     );
    
    maxEpochs = 80;
    miniBatchSize = 128;
    
    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'GradientThreshold', 1, ...
        'Verbose',false ...
        );
    
    net = trainNetwork(xTrain, yTrain, layers, options);
    
    yPred_train = classify(net, xTrain, 'MiniBatchSize', miniBatchSize);
    acc_train = sum(yPred_train == yTrain)./numel(yTrain);
    
    yPred_val = classify(net, xVal, 'MiniBatchSize', miniBatchSize);
    acc_val = sum(yPred_val == yVal)./numel(yVal);
    
    yPred_test = classify(net, xTest, 'MiniBatchSize', miniBatchSize);
    acc_test = sum(yPred_test == yTest)./numel(yTest);
    ACC_TEST = cat(1, ACC_TEST, acc_test);
    
    disp(round([acc_train, acc_val, acc_test] * 100));
    
end

disp([mean(ACC_TEST), std(ACC_TEST)] * 100);
