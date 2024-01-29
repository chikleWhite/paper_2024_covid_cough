%% Hyper-parameters tuning - CRNN - covid-19 classification - cough audio signals

close all; clc; clear;

% set scores parameters
scores = {};
scores(1, :) = [ ...
    {'win_len overlap_win'} {'overlap_spectrograms'} {'spectrum_type'} {'win_norm'} {'filter_bank_norm'} ...
    {'fc layer cnn'} {'outputSize'} ...
    {'numHiddenUnits'} {'drop_prob'} ...
    {'InitialLearnRate'} {'L2Regularization'} {'LearnRateDropFactor'} ...
    {'val_ACC'}, {'val_UAR'}, {'val_F1'} ...
    ];
disp(scores); % display names of parameters

% load relevent layers from pre-traind CNN network (yamnet/vggish)
net_name = "yamnet"; % "yamnet" "vggish"
lgraph_cnn = load_cnn_lgraph(net_name);

% load data
% split to train/test (0.8/0.2 ratio) and keep only train set
shuffle_dataset = "no"; % choose if to shuffle the datasets
[ads, ~] = load_datasets_train_test(shuffle_dataset);

% win & spectrogram length/overlap info
win_lens = [128 256];
overlap_coeff_wins = [0.5 0.75];
overlap_coeff_spectrogramss = [0.75 0.8 0.85];

% win & spectrogram normalization info
spectrum_types = ["power" "magnitude"]; % "power" "magnitude"
win_norms = [true false]; % true false
filter_bank_norms = ["bandwidth" "none"]; % "area" "bandwidth" "none"

% CNN
add_fc_layer_yes_no = ["yes" "no"];
outputSizes = [128 256 512];

% LSTM
numHiddenUnitss = [256 512 1024 2056];
drop_probs = [0.1 0.2 0.3 0.4]; % 0.05 0.1 0.15 0.2 0.25

MaxEpochs = 30;
miniBatchSizes = 4;
InitialLearnRates = [0.00003 0.0001 0.0003 0.001];
L2Regularizations = [0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03];
LearnRateSchedules = "piecewise"; % "none" "piecewise"
LearnRateDropPeriods = 1; % 1 2
LearnRateDropFactors = [0.3 0.7 1];
ValidationPatience = 5;
SequencePaddingDirections = "right"; % "right" "left"
SequenceLength = "longest"; % "longest" (default) | "shortest" | positive integer

num_iters = 100;

tic;

for iter = 1 : num_iters
    
    % win length & and overlap coeff
    rand_indx = randi(length(win_lens), 1);
    win_info.win_len = win_lens(rand_indx);
    win_info.overlap_coeff_win = overlap_coeff_wins(rand_indx);
    
    % overlap coeff spectrograms
    rand_indx = randi(length(overlap_coeff_spectrogramss), 1);
    win_info.overlap_coeff_spectrograms = overlap_coeff_spectrogramss(rand_indx);
    
    % spectrum type
    rand_indx = randi(length(spectrum_types), 1);
    spectrum_type = spectrum_types(rand_indx);
    
    % win norm
    rand_indx = randi(length(win_norms), 1);
    win_norm = win_norms(rand_indx);
    
    % filte bank norm
    rand_indx = randi(length(filter_bank_norms), 1);
    filter_bank_norm = filter_bank_norms(rand_indx);
    
    % extract window, fft and segment info
    win_info = extrct_win_info(win_info);
    
    afe = audioFeatureExtractor(...
        'SampleRate',       win_info.fs, ...
        'Window',           win_info.win, ...
        'OverlapLength',    win_info.numSamplesOverlapBetweenWin, ...
        'melSpectrum',      true ...
        );
    
    setExtractorParams( ...
        afe,                        'melSpectrum', ...
        'SpectrumType',             spectrum_type, ... % 'power' (default) | 'magnitude'
        'NumBands',                 win_info.num_bands, ...
        "WindowNormalization",      win_norm, ... % true (default) | false
        'FilterBankNormalization',  filter_bank_norm ... % 'bandwidth' (default) | 'area' | 'none'
        );
    
    % cnn - add fully connected layer?
    rand_indx = randi(length(add_fc_layer_yes_no), 1);
    add_fc_layer = add_fc_layer_yes_no(rand_indx);
    
    % cnn - size of output from fully connected layer
    rand_indx = randi(length(outputSizes), 1);
    outputSize = outputSizes(rand_indx);
    
    % lstm - num Hidden Units
    rand_indx = randi(length(numHiddenUnitss), 1);
    numHiddenUnits = numHiddenUnitss(rand_indx);
    
    % lstm - dropout probability
    rand_indx = randi(length(drop_probs), 1);
    drop_prob = drop_probs(rand_indx);
    
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
    
    % Sequence Padding Direction
    rand_indx = randi(length(SequencePaddingDirections), 1);
    SequencePaddingDirection = SequencePaddingDirections(rand_indx);
    
    % add fully connected layer to yamnet
    if net_name == "yamnet" && add_fc_layer == "yes"
        
        orig_cnn_last_layer_name = string(lgraph_cnn.Layers(end, 1).Name);
        layers = [
            fullyConnectedLayer(outputSize, 'Name', 'fc_cnn_last')
            batchNormalizationLayer('Name', 'L_cnn_last')
            reluLayer('Name', 'activation_cnn_last')
            ];
        
        % add and connect extra layers (fc, BN, relu) to end of yamnet
        lgraph_cnn = addLayers(lgraph_cnn, layers);
        lgraph_cnn = connectLayers(lgraph_cnn, orig_cnn_last_layer_name, 'fc_cnn_last');
    end
    
    % pre-allocation for val results from CV
    k = 5;
    val_acc = zeros(1, k);
    val_uar = zeros(1, k);
    val_f1 = zeros(1, k);
    
    % loop k-fold times for CV
    for i = 1 : k
        
        % shuffle and seperate to train/validation for CV
        [adsTrain_voca, adsVal_voca] = splitEachLabel(shuffle(ads.UnderlyingDatastores{1,1}), 0.8, 0.2);
        [adsTrain_coughvid, adsVal_coughvid] = splitEachLabel(shuffle(ads.UnderlyingDatastores{1,2}), 0.8, 0.2);
        adsTrain = combine(adsTrain_voca, adsTrain_coughvid);
        adsVal = combine(adsVal_voca, adsVal_coughvid);
        
        % preprocess datasets: extract auto-segmentation results from yamnet model
        [FeaturesTrain, trueLabelsTrain, adsSpecsTrain] = ...
            preprocess_classification_CRNN(adsTrain, win_info, afe);
        [FeaturesVal, trueLabelsVal, adsSpecsVal] = ...
            preprocess_classification_CRNN(adsVal, win_info, afe);
        
        % set classes and classes weights
        uniqueLabels = unique(trueLabelsTrain); % ["negative" "positive"]
        numClasses = length(uniqueLabels); % No. classes
        num_train_pos = sum(trueLabelsTrain == "positive");
        num_train_neg = sum(trueLabelsTrain == "negative");
        num_train = num_train_pos + num_train_neg;
        
        % give weight to each class by the other one relative size
        classWeights = [num_train_pos, num_train_neg] / num_train;
        
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

        % add new input layer (sequence input for lstm) and folding layer
        inputSize = [96 64]; % input size needs to be the same as for CNN
        layers = [
            sequenceInputLayer([inputSize 1], 'Name', 'input') % sequence input for lstm
            sequenceFoldingLayer('Name', 'fold') % change structure for CNN
            ];
        
        % connect new input layers to input of CNN
        lgraph = addLayers(lgraph_cnn, layers);
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
            'MaxEpochs', MaxEpochs, ...
            'MiniBatchSize', miniBatchSize, ...
            'Shuffle', 'every-epoch', ... % 'once' (default) | 'never' | 'every-epoch'
            'ValidationData', {FeaturesVal, trueLabelsVal}, ...
            'ValidationFrequency', validationFrequency, ...
            'InitialLearnRate', InitialLearnRate, ...
            'LearnRateSchedule', LearnRateSchedule, ... % 'none' (default) | 'piecewise'
            'LearnRateDropPeriod', LearnRateDropPeriod, ...
            'LearnRateDropFactor', LearnRateDropFactor, ...
            'L2Regularization', L2Regularization, ...
            'ValidationPatience', ValidationPatience, ...
            'SequencePaddingDirection', SequencePaddingDirection, ...
            'SequenceLength', SequenceLength ...% "longest" (default) | "shortest" | positive integer
            );
        
        [net_classification_CRNN, netInfo] = trainNetwork(FeaturesTrain, trueLabelsTrain, lgraph, options);
        
        % predict on validation set
        miniBatchSize_predict = 1;
        predLabelsVal = classify(net_classification_CRNN, FeaturesVal, 'MiniBatchSize', miniBatchSize_predict);
        
        % TP TN FP FN
        TP = sum(predLabelsVal == trueLabelsVal & trueLabelsVal == 'positive');
        TN = sum(predLabelsVal == trueLabelsVal & trueLabelsVal == 'negative');
        FP = sum(predLabelsVal ~= trueLabelsVal & trueLabelsVal == 'negative');
        FN = sum(predLabelsVal ~= trueLabelsVal & trueLabelsVal == 'positive');
        
        % calculate scores per segment
        val_acc(i) =       round((TP + TN) / (TP + TN + FP + FN), 4) * 100;
        val_uar(i) =       round((TP / (TP + FN) + TN / (TN + FP)) / 2, 4) * 100;
        val_f1(i) =  round(TP ./ (TP + (FP + FN)/2), 4) * 100;
    end
    
    val_ACC = mean(val_acc);
    val_UAR = mean(val_uar);
    val_F1 = mean(val_f1);
    
    scores(iter + 1, :) = [ ...
        {[win_info.win_len win_info.overlap_coeff_win]} {win_info.overlap_coeff_spectrograms} {spectrum_type} {win_norm} {filter_bank_norm} ...
        {numHiddenUnits} {drop_prob}, ...
        {InitialLearnRate} {L2Regularization} {LearnRateDropFactor} ...
        {val_ACC} {val_UAR} {val_F1} ...
        ];
    
    % remove previous text from command window
    clc;
    
    % show scores
    disp(scores);
    
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

writecell( ...
    scores, ...
    ['covid classification results ', char(net_name), '/scores classification CRNN hyperparameters tuning.csv'] , ...
    'WriteMode' , 'overwrite' ...
    );
