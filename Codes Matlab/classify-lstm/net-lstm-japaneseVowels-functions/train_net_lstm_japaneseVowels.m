%% Train a deep learning LSTM network for sequence-to-label classification.

% remove previous data
close all; clc; clear;

% Load the Japanese Vowels data set as described in [1] and [2].
% XTrain is a cell array containing 270 sequences of varying length with 12 features corresponding to LPC cepstrum coefficients.
% Y is a categorical vector of labels 1,2,...,9.
% The entries in XTrain are matrices with 12 rows (one row for each feature),
% and a varying number of columns (one column for each time step).

[XTrain, YTrain] = japaneseVowelsTrainData;
[XTest, YTest] = japaneseVowelsTestData;

% combine train & test to 1 single dataset
X = [XTrain ; XTest];
Y = [YTrain ; YTest];

% Define the LSTM network architecture.
% Specify the input size as 12 (the number of features of the input data).
% Specify an LSTM layer to have 100 hidden units and to output the last element of the sequence.
% Finally, specify nine classes by including a fully connected layer of size 9,
% followed by a softmax layer and a classification layer.

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

% Specify the training options.
% Specify the solver as 'adam' and 'GradientThreshold' as 1.
% Set the mini-batch size to 27 and set the maximum number of epochs to 70.

maxEpochs = 80;
miniBatchSize = 128;

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'GradientThreshold', 1, ...
    'Verbose',false, ...
    'Plots','training-progress' ...
    );

% Train the LSTM network with the specified training options.
net_lstm_japaneseVowels = trainNetwork(X, Y, layers, options);

% Change current folder
newFolder = "../";
oldFolder = cd(newFolder);

% save network
save net_lstm_japaneseVowels net_lstm_japaneseVowels

% Change the current folder back to the original folder
cd(oldFolder);
