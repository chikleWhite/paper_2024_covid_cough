function [adsTrain, adsVal, adsTest] = load_datasets_train_val_test(dataset_name, split_prcnts, shuffle_dataset)

% pre-allocation
adsTrain = [];
adsVal = [];


% number of datasets
if dataset_name == "all"
    num_datasets = 2;
else
    num_datasets = 1;
end

% percentages of (train, val, test) for splitEachLabel
TrainVal_prcnt = split_prcnts(1) + split_prcnts(2);
Test_prcnt = split_prcnts(3);
Train_prcnt = split_prcnts(1) / TrainVal_prcnt;
Val_prcnt = 1 - Train_prcnt;

% load voca dataset
ds_name = "voca";
folder_name = "../../../DataBase/" + ds_name + "/" + "audioDataFolder";
ads_voca = audioDatastore(folder_name, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

% load coughvid dataset
ds_name = "coughvid";
folder_name = "../../../DataBase/" + ds_name + "/" + "audioDataFolder";
ads_coughvid = audioDatastore(folder_name, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

% shuffle datasets (train, val and test)
if shuffle_dataset == "yes"
    ads_voca = shuffle(ads_voca); % shuffle voca dataset
    ads_coughvid = shuffle(ads_coughvid); % shuffle coughvid dataset
end

% split voca dataset to (train + val, test)
[adsTrainVal_voca, adsTest_voca] = ...
    splitEachLabel(ads_voca, TrainVal_prcnt, Test_prcnt);

% split coughvid dataset to (train + val, test)
[adsTrainVal_coughvid, adsTest_coughvid] = ...
    splitEachLabel(ads_coughvid, TrainVal_prcnt, Test_prcnt);

% combine datasets
switch dataset_name
    
    case "all" % both voca and coughvid datasets
        
        adsTrainVal =   {adsTrainVal_voca ; adsTrainVal_coughvid};
        adsTest =       {adsTest_voca ; adsTest_coughvid};
        
    case  "voca" % only voca dataset
        
        adsTrainVal =   {adsTrainVal_voca};
        adsTest =       {adsTest_voca};
        
    case "coughvid" % only coughvid dataset
        
        adsTrainVal =   {adsTrainVal_coughvid};
        adsTest =       {adsTest_coughvid};
end

% split sets to train and val -> combine again
switch shuffle_dataset
    
    case "no" % split, don't shuffle
        
        for i = 1 : num_datasets
            [adsTrain_set, adsVal_set] = splitEachLabel(adsTrainVal{i}, Train_prcnt, Val_prcnt);
            adsTrain =  cat(2, adsTrain, adsTrain_set);
            adsVal =    cat(2, adsTrain, adsVal_set);
        end
        
    case "train_val" % split, shuffle train and val sets
        
       for i = 1 : num_datasets
            [adsTrain_set, adsVal_set] = splitEachLabel(shuffle(adsTrainVal{i}), Train_prcnt, Val_prcnt);
            adsTrain =  cat(2, adsTrain, adsTrain_set);
            adsVal =    cat(2, adsTrain, adsVal_set);
        end
end

end

