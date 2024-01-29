%% Hyper-parameters tuning - CNN - covid-19 classification - cough audio signals

%% choose gpu

% remove previous data
close all; clc; clear;

% choose gpu
gpu_selected = "0"; % "0" | "1" | "no"
select_gpu(gpu_selected);

%% run RCNN

% datasets loading and spliting info
dataset_name = "all_ds"; % "all_ds" | "voca" | "voca_old" | "coughvid"
split_prcnts = [0.6 0.2 0.2]; % size of train val test in [%]
shuffle_datasets = "train_val"; % "all" | "train_val" | "no"

% load relevent layers from pre-traind CNN networks
net_cnn_names = ["yamnet" "vggish" "openl3"]; % "yamnet" | "vggish" | "openl3"
freeze_layers = "no";
lgraph_cnn_yamnet = load_cnn_lgraph("yamnet", freeze_layers);
lgraph_cnn_vggish = load_cnn_lgraph("vggish", freeze_layers);
lgraph_cnn_openl3 = load_cnn_lgraph("openl3", freeze_layers);
lgraph_cnns = [{lgraph_cnn_yamnet} {lgraph_cnn_vggish} {lgraph_cnn_openl3}];

% No. CV
k = 5;

% win & spectrogram length/overlap info
win_lens = [128 256]; % [128 256]
overlap_coeff_wins = [0.5 0.75]; % [0.5 0.75]
overlap_coeff_spectrogramss = [0.75 0.8 0.85 0.9]; % [0.75 0.8 0.85 0.9]

% win & spectrogram normalization info
spectrum_types = ["power" "magnitude"]; % "power" "magnitude"
win_norms = [true false]; % true false
filter_bank_norms = "none"; % "area" "bandwidth" "none"

MaxEpochs = 30;
miniBatchSizes = [16 32 64]; % [4 8]
InitialLearnRates = [0.0001 0.0003 0.001 0.003 0.01];
L2Regularizations = [0.0001 0.0003 0.001 0.003 0.01];
LearnRateSchedules = "piecewise"; % "none" "piecewise"
LearnRateDropPeriods = 1; % 1 2
LearnRateDropFactors = [0.7 0.9 0.99];
ValidationPatiences = 5;
checkpointPath = "checkpoint"; % save net in folder after each epoch

tpr_thershold = 0.6;
num_iters = 500;
    
% set scores parameters
scores(1, :) = [ ...
    {'shuffle_ds'} {'split_prcnts'} ...
    {'net_cnn_name'} {'freeze_layers'} ...
    {'win_len'} {'overlap_win'} {'overlap_spectrograms'} ...
    {'spectrum_type'} {'win_norm'} {'filter_bank_norm'} ...
    {'miniBatchSize'} {'InitialLearnRate'} {'L2Regularization'} {'LearnRateDropFactor'} {'ValidationPatience'} ...
    {'acc'} {'uar'} {'F1'} {'sensitivity'} {'PPV'} {'specificity'} ...
    {'loss_train'} {'loss_val'} ...
    ];
disp(scores(1, end-7 : end)); % display names of parameters

tic;

for iter = 1 : num_iters
    
    % cnn model
    rand_indx = randi(length(lgraph_cnns), 1);
    lgraph_cnn = lgraph_cnns{rand_indx};
    net_cnn_name = net_cnn_names(rand_indx);
    
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
    
    % extract window, fft and segment info
    win_info = extrct_win_info(win_info);
    
    % Streamline audio feature extraction (mel-spectrogram)
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
    
    % pre-allocation for val results from CV
    val_acc =           zeros(1, k);
    val_uar =           zeros(1, k);
    val_f1 =            zeros(1, k);
    val_sensitivity =   zeros(1, k);
    val_ppv =           zeros(1, k);
    val_specificity =   zeros(1, k);
    val_loss =          zeros(1, k);
    train_loss =        zeros(1, k);
    
    % loop k-fold times for CV
    for i = 1 : k
        
        % load data and split to train val test
        [adsTrain, adsVal, adsTest] = load_split_shuffle_datasets(dataset_name, split_prcnts, shuffle_datasets);
        
        % preprocess datasets: extract auto-segmentation results from yamnet model
        [FeaturesPerSegTrain, trueLabelsPerSegTrain, adsSpecsTrain] = ...
            preprocess_classification_CNN(adsTrain, win_info, afe);
        [FeaturesPerSegVal, trueLabelsPerSegVal, adsSpecsVal] = ...
            preprocess_classification_CNN(adsVal, win_info, afe);
        [FeaturesPerSegTest, trueLabelsPerSegTest, adsSpecsTest] = ...
            preprocess_classification_CNN(adsTest, win_info, afe);
        
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
        validationFrequency = floor(size(FeaturesPerSegTrain, 4) / miniBatchSize);
        options = trainingOptions( ...
            'adam', ...
            'Plots', 'none', ... % 'none' (default) | 'training-progress'
            'Verbose', false, ... % 1 (true) (default) | 0 (false)
            'MaxEpochs', MaxEpochs, ...
            'MiniBatchSize', miniBatchSize, ...
            'Shuffle', 'every-epoch', ... % 'once' (default) | 'never' | 'every-epoch'
            ...
            'ValidationData', {FeaturesPerSegVal, trueLabelsPerSegVal}, ...
            'ValidationFrequency', validationFrequency, ...
            'InitialLearnRate', InitialLearnRate, ...
            'LearnRateSchedule', LearnRateSchedule, ... % 'none' (default) | 'piecewise'
            'LearnRateDropPeriod', LearnRateDropPeriod, ...
            'LearnRateDropFactor', LearnRateDropFactor, ...
            'L2Regularization', L2Regularization, ...
            'ValidationPatience', ValidationPatience, ...
            ...
            'CheckpointPath', checkpointPath, ...
            'BatchNormalizationStatistics', 'moving' ... % 'population' (default) | 'moving'
            );
        
        % remove saved nets from previous training
        delete(checkpointPath + "/*");
        
        % train network
        [~, net_info] = trainNetwork(FeaturesPerSegTrain, trueLabelsPerSegTrain, lgraph, options);
        
        % load net (out of all nets yelded from current training) from epoch with best result (lowest validation error)
        net_classification_CNN = extract_net_from_checkpoint(net_info, checkpointPath);
        
        % predict on validation set
        positive_indx = find(string(uniqueLabels) == "positive");
        predLabelsPerSegVal = predict(net_classification_CNN, FeaturesPerSegVal, 'MiniBatchSize', miniBatchSize);
        predLabelsPerSegVal = predLabelsPerSegVal(:, positive_indx);
        
        % change predictions and true labels from per spectrogram to per subject
        [predLabelsPerSubjVal, trueLabelsPerSubjVal] = perSegment2perSubject(predLabelsPerSegVal, trueLabelsPerSegVal, adsSpecsVal);
        
        % find best ROC thrshold on val set
        
        % change true labels from name of classes to logical (0/1) - on val set
        lbls_true = (string(trueLabelsPerSubjVal) == "positive")';
        
        % pred labels from val set
        lbls_pred = predLabelsPerSubjVal';
        
        % get ROC info
        [tpr, fpr, roc_thresholds] = roc(lbls_true, lbls_pred); % ROC values (TPR = sensetivity, FPR = 1 - specificity)
        
        % find optimal threshold based on minimum distance from point (1,0) ->
        % tpr = 1, fpr = 0 & sensitivity >= tpr_thershold
        indx_low_sensitivity = tpr < tpr_thershold;
        dist_from_opt_point = ((1 - tpr).^2 + (0 - fpr).^2).^0.5;
        dist_from_opt_point(indx_low_sensitivity) = 1;
        [~, indx_min_dist] = min(dist_from_opt_point);
        roc_threshold = roc_thresholds(indx_min_dist);
        
        % convert predictions from probabilities to logical (0/1)
        lbls_pred = (predLabelsPerSubjVal >= roc_threshold);
        
        % convert predictions from logical (0/1) to categorical (names of classes)
        lbls_pred = categorical(lbls_pred, [0 1], {'negative' 'positive'});
        
        % convert labels (predictions and true) from categorical to string
        lbls_pred = string(lbls_pred);
        lbls_true = string(trueLabelsPerSubjVal);
        
        % TP TN FP FN
        TP = sum(lbls_pred == lbls_true & lbls_true == "positive");
        TN = sum(lbls_pred == lbls_true & lbls_true == "negative");
        FP = sum(lbls_pred ~= lbls_true & lbls_true == "negative");
        FN = sum(lbls_pred ~= lbls_true & lbls_true == "positive");
        
        % calculate scores per cross-val
        val_acc(i)      = round((TP + TN) / (TP + TN + FP + FN), 4) * 100;
        val_uar(i)      = round((TP / (TP + FN) + TN / (TN + FP)) / 2, 4) * 100;
        val_f1(i)       = round(TP ./ (TP + (FP + FN)/2), 4) * 100;
        val_sensitivity(i)  =   round(TP / (TP + FN), 3) * 100;
        val_ppv(i)          =   round(TP / (TP + FP), 3) * 100;
        val_specificity(i)  =   round(TN / (TN + FP), 3) * 100;
        [val_loss(i), val_loss_indx] = min(net_info.ValidationLoss);
        train_loss(i) = ...
            mean(net_info.TrainingLoss(max(val_loss_indx - validationFrequency + 1, 1) : val_loss_indx));
    end
    
    % take mean results
    val_ACC = mean(val_acc);
    val_UAR = mean(val_uar);
    val_F1 = mean(val_f1);
    val_Sensitivity = mean(val_sensitivity);
    val_PPV = mean(val_ppv);
    val_Specificity = mean(val_specificity);
    val_LOSS = mean(val_loss);
    train_LOSS = mean(train_loss);
    
    scores(iter + 1, :) = [ ...
        {split_prcnts} {shuffle_datasets} ...
        {net_cnn_name} {freeze_layers} ...
        {win_info.win_len} {win_info.overlap_coeff_win} {win_info.overlap_coeff_spectrograms} ...
        {spectrum_type} {win_norm} {filter_bank_norm} ...
        {miniBatchSize} {InitialLearnRate} {L2Regularization} {LearnRateDropFactor} {ValidationPatience} ...
        {val_ACC} {val_UAR} {val_F1} {val_Sensitivity} {val_PPV} {val_Specificity} ...
        {train_LOSS} {val_LOSS} ...
        ];
    
    % save scores in csv file
    writecell( ...
        scores, ...
    ['covid classification results ', char(dataset_name), '/scores classification CNN hyperparameters tuning.csv'], ...
    'WriteMode' , 'overwrite' ...
    );
    
    % remove previous text from command window
    clc;
    
    % show scores
    disp(scores(1 : iter + 1, end-7 : end));
    
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
