%% Hyper-parameters tuning - CRNN - covid-19 classification - cough audio signals

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpu_selected = "0"; % "0" | "1" | "no"
select_gpu(gpu_selected);

%% run CRNN in loop to find best hyper-parameters

% load ds
% [Features, trueLabels, adsSpecs, win_info] = load_ds(); % for 1st time only
load ds;

% shuffle and split data into train val test
shuffle_ds = "TrainVal"; % "yes" | "no" | "TrainVal"
balance_ds = "TrainVal"; % "yes" | "no" | "TrainVal"
split_prcnts = [0.6 0.2 0.2]; % size of train val test in [%]

% load relevent layers from pre-traind CNN network (yamnet/vggish/openl3)
net_cnn_name = "vggish"; % "yamnet" | "vggish" | "openl3"
freeze_layers = "no";
lgraph_cnn = load_cnn_lgraph(net_cnn_name, freeze_layers);

% save graph for adding extra layers each iteration
lgraph_cnn_temp = lgraph_cnn;

% CNN
% add_FC_CNN = ["yes" "no"]; % "yes" "no"
% outputSize_FC_CNN = [256 512 1024];

% LSTM
numHiddenUnitss = [256 512 1024];
drop_probs = [0.01 0.03 0.1 0.3]; % 0.05 0.1 0.15 0.2 0.25
% add_FC_LSTM = ["yes" "no"]; % "yes" "no"
% outputSize_FC_LSTM = [256 512 1024];

MaxEpochs = 20;
miniBatchSizes = [4 8]; % [4 8]
InitialLearnRates = [0.0001 0.0003 0.001 0.003 0.01];
L2Regularizations = [0.0001 0.0003 0.001 0.003 0.01];
LearnRateSchedules = "piecewise"; % "none" "piecewise"
LearnRateDropPeriods = 1; % 1 2
LearnRateDropFactors = [0.7 0.9 0.99];
ValidationPatiences = 10;
SequencePaddingDirections = "right"; % "right" "left"
SequenceLength = "longest"; % "longest" (default) | "shortest" | positive integer
checkpointPath = "checkpoint"; % save net in folder after each epoch

num_iters = 100;

% set scores parameters
scores = cell(num_iters + 1, 23);
scores(1, :) = [ ...
    {'shuffle_ds'} {'balance_ds'} {'split_prcnts'} ...
    {'net_cnn_name'} {'freeze_layers'} ...
    {'numHiddenUnits'} {'drop_prob'} ...
    {'MaxEpochs'} {'miniBatchSize'} {'InitialLearnRate'} {'L2Regularization'} {'LearnRateDropFactor'} {'ValidationPatiences'} ...
    {'acc'} {'uar'} {'F1'} {'sensitivity'} {'PPV'} {'specificity'} ...
    {'loss_train_best'} {'loss_val_best'} {'loss_train_end'} {'loss_val_end'} ...
    ];

disp(scores(1, end - 9 : end)); % display names of parameters

tic;

for iter = 1 : num_iters
    
%     % cnn - size of output from fully connected layer
%     rand_indx = randi(length(add_FC_CNN), 1);
%     add_fc_cnn = add_FC_CNN(rand_indx);
%     
%     % cnn - size of output from fully connected layer
%     rand_indx = randi(length(outputSize_FC_CNN), 1);
%     outputSize_fc_cnn = outputSize_FC_CNN(rand_indx);
    
    % lstm - num Hidden Units
    rand_indx = randi(length(numHiddenUnitss), 1);
    numHiddenUnits = numHiddenUnitss(rand_indx);
    
    % lstm - dropout probability
    rand_indx = randi(length(drop_probs), 1);
    drop_prob = drop_probs(rand_indx);
    
%     % lstm - size of output from fully connected layer
%     rand_indx = randi(length(add_FC_LSTM), 1);
%     add_fc_lstm = add_FC_LSTM(rand_indx);
%     
%     % lstm - size of output from fully connected layer
%     rand_indx = randi(length(outputSize_FC_LSTM), 1);
%     outputSize_fc_lstm = outputSize_FC_LSTM(rand_indx);
    
    % mini Batch Size
    rand_indx = randi(length(miniBatchSizes), 1);
    miniBatchSize = miniBatchSizes(rand_indx);
    
    % InitialLearn Rate
    rand_indx = randi(length(InitialLearnRates), 1);
    InitialLearnRate = InitialLearnRates(rand_indx);
    
    % L2 Regularization
    rand_indx = randi(length(L2Regularizations), 1);
    L2Regularization = L2Regularizations(rand_indx);
    
    % Learn Rate Schedule
    rand_indx = randi(length(LearnRateSchedules), 1);
    LearnRateSchedule = LearnRateSchedules(rand_indx);
    
    % Learn Rate Drop Period
    rand_indx = randi(length(LearnRateDropPeriods), 1);
    LearnRateDropPeriod = LearnRateDropPeriods(rand_indx);
    
    % Learn Rate Drop Factor
    rand_indx = randi(length(LearnRateDropFactors), 1);
    LearnRateDropFactor = LearnRateDropFactors(rand_indx);
    
    % Validation Patiences
    rand_indx = randi(length(ValidationPatiences), 1);
    ValidationPatience = ValidationPatiences(rand_indx);
    
    % Sequence Padding Direction
    rand_indx = randi(length(SequencePaddingDirections), 1);
    SequencePaddingDirection = SequencePaddingDirections(rand_indx);
    
    % shuffle and split data into train val test
    [FeaturesTrain, trueLabelsTrain, FeaturesVal, trueLabelsVal, FeaturesTest, trueLabelsTest] = ...
        split_shuffle_ds(Features, trueLabels, shuffle_ds, balance_ds, split_prcnts);
    
    % set classes and classes weights
    uniqueLabels = unique(trueLabelsTrain); % ["negative" "positive"]
    numClasses = length(uniqueLabels); % No. classes
    num_train_pos = sum(trueLabelsTrain == "positive");
    num_train_neg = sum(trueLabelsTrain == "negative");
    num_train = num_train_pos + num_train_neg;
    
    % give weight to each class by the other one relative size
%     classWeights = [num_train_pos, num_train_neg] / num_train;
    classWeights = num_train ./ ([num_train_neg num_train_pos] * numClasses);
    
%     % add fully connected layer to cnn
%     
%     switch net_cnn_name
%         
%         case "vggish"
%             
%             layers = [
%                 fullyConnectedLayer(outputSize_fc_cnn, 'Name', 'fc_cnn_last')
%                 reluLayer('Name', 'activation_cnn_last')
%                 ];
%             
%         case "yamnet"
%             
%             layers = [
%                 fullyConnectedLayer(outputSize_fc_cnn, 'Name', 'fc_cnn_last')
%                 batchNormalizationLayer('Name', 'BN_cnn_last')
%                 reluLayer('Name', 'activation_cnn_last')
%                 ];
%     end
%     
%     % add and connect extra layers (fc, relu) to end of cnn
%     lgraph_cnn = addLayers(lgraph_cnn_temp, layers);
%     lgraph_cnn = connectLayers( ...
%         lgraph_cnn, ...
%         string(lgraph_cnn_temp.Layers(end, 1).Name), ...
%         string(layers(1, 1).Name));
    
%     % create LSTM network
%     lstmLayers = [
%         bilstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm')
%         dropoutLayer(drop_prob, 'Name', 'drop')
%         fullyConnectedLayer(outputSize_fc_lstm, 'Name', 'fc_lstm1')
%         reluLayer('Name', 'relu_lstm1')
%         fullyConnectedLayer(numClasses, 'Name', 'fc')
%         softmaxLayer('Name', 'softmax')
%         classificationLayer("Name", "classification", "Classes", uniqueLabels, "ClassWeights", classWeights)
%         ];

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
    lgraph_crnn = layerGraph;
    
    % add new input layer (sequence input for lstm) and folding layer
    inputSize = [96 64]; % input size needs to be the same as for CNN
    layers = [
        sequenceInputLayer([inputSize 1], 'Name', 'input') % sequence input for lstm
        sequenceFoldingLayer('Name', 'fold') % change structure for CNN
        ];
    
    % connect new input layers to input of CNN
    lgraph_crnn = addLayers(lgraph_crnn, layers);
    lgraph_crnn = addLayers(lgraph_crnn, lgraph_cnn.Layers);
    lgraph_crnn = connectLayers(lgraph_crnn, "fold/out", string(lgraph_cnn.Layers(1, 1).Name));
    
    % add unfolding and lstm layers
    layers = [
        sequenceUnfoldingLayer('Name', 'unfold') % change structure back for lstm
        flattenLayer('Name', 'flatten') % change dim from [1, 1, No. output features] to [No. output features, 1]
        lstmLayers
        ];
    
    % connect unfolding and lstm layers to output of CNN
    lgraph_crnn = addLayers(lgraph_crnn, layers);
    lgraph_crnn = connectLayers(lgraph_crnn, string(lgraph_cnn.Layers(end, 1).Name), "unfold/in");
    
    % add parallel connection between folding & unfolding layers (needed to restore the sequence structure)
    lgraph_crnn = connectLayers(lgraph_crnn, "fold/miniBatchSize", "unfold/miniBatchSize");
    
    % specify training options
    validationFrequency = floor(numel(trueLabelsTrain)/miniBatchSize);
    options = trainingOptions( ...
        'adam', ...
        'Plots', 'none', ... % 'none' (default) | 'training-progress'
        'Verbose', false, ... % 1 (true) (default) | 0 (false)
        'MaxEpochs', MaxEpochs, ...
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
    [net_end_epoch, net_info] = trainNetwork(FeaturesTrain, trueLabelsTrain, lgraph_crnn, options);
    
    % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
    net_best_epoch = extract_net_from_checkpoint(net_info, checkpointPath);
    
    % predict - best epoch
    miniBatchSize_predict = 1;
    positive_indx = find(string(uniqueLabels) == "positive");
    predLabelsVal = predict(net_best_epoch, FeaturesVal, 'MiniBatchSize', miniBatchSize_predict);
    predLabelsVal = predLabelsVal(:, positive_indx);
    
    
    % find scores for val at best epoch
    
    % get ROC info (TPR = sensetivity, FPR = 1 - specificity)
    [tpr, fpr, roc_thresholds] = roc((string(trueLabelsVal) == "positive")', predLabelsVal');
    
    % find optimal threshold based on minimum distance from point (1,0) ->
    % tpr = 1, fpr = 0
    tpr_thershold = 0.6;
    indx_low_sensitivity = tpr < tpr_thershold;
    dist_from_opt_point = ((1 - tpr).^2 + (0 - fpr).^2).^0.5;
    dist_from_opt_point(indx_low_sensitivity) = 1;
    [~, indx_min_dist] = min(dist_from_opt_point);
    roc_threshold = roc_thresholds(indx_min_dist);
    
    % convert predictions from probabilities to logical (0/1)
    lbls_pred = (predLabelsVal >= roc_threshold);
    
    % convert predictions from logical (0/1) to categorical (names of classes)
    lbls_pred = categorical(lbls_pred, [0 1], {'negative' 'positive'});
    
    % convert labels (predictions and true) from categorical to string
    lbls_pred = string(lbls_pred);
    lbls_true = string(trueLabelsVal);
    
    % TP TN FP FN
    TP = sum(lbls_pred == lbls_true & lbls_true == "positive");
    TN = sum(lbls_pred == lbls_true & lbls_true == "negative");
    FP = sum(lbls_pred ~= lbls_true & lbls_true == "negative");
    FN = sum(lbls_pred ~= lbls_true & lbls_true == "positive");
    
    % calculate scores
    acc         =   round((TP + TN) / (TP + TN + FP + FN), 3) * 100;
    uar         =   round((TP / (TP + FN) + TN / (TN + FP)) / 2, 3) * 100;
    F1          =   round(TP ./ (TP + (FP + FN)/2), 3) * 100;
    sensitivity =   round(TP / (TP + FN), 3) * 100;
    PPV         =   round(TP / (TP + FP), 3) * 100;
    specificity =   round(TN / (TN + FP), 3) * 100;
    
    % calculate loss at best epoch
    [loss_val_best, loss_indx] = min(net_info.ValidationLoss);
    loss_train_best = ...
        mean(net_info.TrainingLoss(max(loss_indx - validationFrequency + 1, 1) : loss_indx));
    
    % calculate loss at last epoch
    loss_val_end = net_info.ValidationLoss(end);
    loss_train_end = ...
        mean(net_info.TrainingLoss(max(end - validationFrequency + 1, 1) : end));
    
    scores(iter + 1, :) = [ ...
    {shuffle_ds} {balance_ds} {split_prcnts} ...
    {net_cnn_name} {freeze_layers} ...
    {numHiddenUnits} {drop_prob} ...
    {MaxEpochs} {miniBatchSize} {InitialLearnRate} {L2Regularization} {LearnRateDropFactor} {ValidationPatiences} ...
    {acc} {uar} {F1} {sensitivity} {PPV} {specificity} ...
    {loss_train_best} {loss_val_best} {loss_train_end} {loss_val_end} ...
    ];
    
    % save scores in file
    writecell( ...
        scores, ...
        "covid classification results " + net_cnn_name + "/scores classification CRNN hyperparameters tuning.csv" , ...
        'WriteMode' , 'overwrite' ...
        );
    
    % remove previous text from command window
    clc;
    
    % show scores
    disp(scores(1 : iter + 1, end - 9 : end));
    
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

% deselects the GPU device and clears its memory
gpuDevice([]);
