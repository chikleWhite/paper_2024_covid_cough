function [Features_Train, trueLabels_Train, Features_Val, trueLabels_Val, Features_Test, trueLabels_Test] = ...
    split_shuffle_ds(Features, trueLabels, shuffle_ds, balance_ds, split_prcnts)

% percentages of train, val, test
TrainVal_prcnt = split_prcnts(1) + split_prcnts(2);
Train_prcnt = split_prcnts(1) / TrainVal_prcnt;

% split ds into positive negative
Features_pos = Features(trueLabels == "positive");
Features_neg = Features(trueLabels == "negative");
trueLabels_pos = trueLabels(trueLabels == "positive");
trueLabels_neg = trueLabels(trueLabels == "negative");

% length of pos/neg ds
len_pos = length(trueLabels_pos);
len_neg = length(trueLabels_neg);

% shuffle dataset (train, val and test)
if shuffle_ds == "yes"
    
    % new random indices
    indx_rand_pos = randperm(len_pos);
    indx_rand_neg = randperm(len_neg);
    
    % shuffle pos ds
    Features_pos = Features_pos(indx_rand_pos);
    trueLabels_pos = trueLabels_pos(indx_rand_pos);
    
    % shuffle neg ds
    Features_neg = Features_neg(indx_rand_neg);
    trueLabels_neg = trueLabels_neg(indx_rand_neg);
end

% balance dataset (train, val and test)
% reduce No. negative subject to be equal to positive
if balance_ds == "yes"
    
    % get indices to remove
    num_neg2remove = len_neg - len_pos;
    
    % remove indices
    Features_neg(end - num_neg2remove + 1 : end) = [];
    trueLabels_neg(end - num_neg2remove + 1 : end) = [];
    
    % update No. negative subjects
    len_neg = len_neg - num_neg2remove;
end

% split dataset to (train + val, test)

% lengths
len_pos_TrainVal = round(TrainVal_prcnt * len_pos);
len_neg_TrainVal = round(TrainVal_prcnt * len_neg);

% indices
indx_pos_TrainVal   = 1 : len_pos_TrainVal;
indx_pos_Test       = len_pos_TrainVal + 1 : len_pos;
indx_neg_TrainVal   = 1 : len_neg_TrainVal;
indx_neg_Test       = len_neg_TrainVal + 1 : len_neg;

% split positive
Features_pos_TrainVal   = Features_pos(indx_pos_TrainVal);
trueLabels_pos_TrainVal = trueLabels_pos(indx_pos_TrainVal);
Features_pos_Test       = Features_pos(indx_pos_Test);
trueLabels_pos_Test     = trueLabels_pos(indx_pos_Test);

% split negative
Features_neg_TrainVal   = Features_neg(indx_neg_TrainVal);
trueLabels_neg_TrainVal = trueLabels_neg(indx_neg_TrainVal);
Features_neg_Test       = Features_neg(indx_neg_Test);
trueLabels_neg_Test     = trueLabels_neg(indx_neg_Test);


% length of TrainVal pos/neg ds
len_pos_TrainVal = length(trueLabels_pos_TrainVal);
len_neg_TrainVal = length(trueLabels_neg_TrainVal);

% shuffle dataset (train and val)
if shuffle_ds == "TrainVal"
    
    % new random indices
    indx_rand_pos_TrainVal = randperm(len_pos_TrainVal);
    indx_rand_neg_TrainVal = randperm(len_neg_TrainVal);
    
    % shuffle pos ds
    Features_pos_TrainVal = Features_pos_TrainVal(indx_rand_pos_TrainVal);
    trueLabels_pos_TrainVal = trueLabels_pos_TrainVal(indx_rand_pos_TrainVal);
    
    % shuffle neg ds
    Features_neg_TrainVal = Features_neg_TrainVal(indx_rand_neg_TrainVal);
    trueLabels_neg_TrainVal = trueLabels_neg_TrainVal(indx_rand_neg_TrainVal);
end

% balance dataset (train + val)
% reduce No. negative subject to be equal to positive
if balance_ds == "TrainVal"
    
    % get indices to remove
    num_neg2remove = len_neg_TrainVal - len_pos_TrainVal;
    
    % remove indices
    Features_neg_TrainVal(end - num_neg2remove + 1 : end) = [];
    trueLabels_neg_TrainVal(end - num_neg2remove + 1 : end) = [];
    
    % update No. negative subjects
    len_neg_TrainVal = len_neg_TrainVal - num_neg2remove;
end

% split dataset to (train, val, test)

% lengths
len_pos_Train = round(Train_prcnt * len_pos_TrainVal);
len_neg_Train = round(Train_prcnt * len_neg_TrainVal);

% indices
indx_pos_Train   = 1 : len_pos_Train;
indx_pos_Val     = len_pos_Train + 1 : len_pos_TrainVal;
indx_neg_Train   = 1 : len_neg_Train;
indx_neg_Val     = len_neg_Train + 1 : len_neg_TrainVal;

% split positive
Features_pos_Train      = Features_pos_TrainVal(indx_pos_Train);
trueLabels_pos_Train    = trueLabels_pos_TrainVal(indx_pos_Train);
Features_pos_Val        = Features_pos_TrainVal(indx_pos_Val);
trueLabels_pos_Val      = trueLabels_pos_TrainVal(indx_pos_Val);

% split negative
Features_neg_Train   = Features_neg_TrainVal(indx_neg_Train);
trueLabels_neg_Train = trueLabels_neg_TrainVal(indx_neg_Train);
Features_neg_Val       = Features_neg_TrainVal(indx_neg_Val);
trueLabels_neg_Val     = trueLabels_neg_TrainVal(indx_neg_Val);


% combine pos/neg

% train
Features_Train = [Features_neg_Train ; Features_pos_Train];
trueLabels_Train = [trueLabels_neg_Train ; trueLabels_pos_Train];

% val
Features_Val = [Features_neg_Val ; Features_pos_Val];
trueLabels_Val = [trueLabels_neg_Val ; trueLabels_pos_Val];

% test
Features_Test = [Features_neg_Test ; Features_pos_Test];
trueLabels_Test = [trueLabels_neg_Test ; trueLabels_pos_Test];

end

