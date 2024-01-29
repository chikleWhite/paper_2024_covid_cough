function [adsTrain, adsVal, adsTest] = load_datasets_train_val_test(shuffle_dataset, dataset_name)

% percentages of (train, val, test) for splitEachLabel
train_prcnt = 0.6;
test_prcnt = 0.2;
val_prcnt = 1 - train_prcnt - test_prcnt;

% load voca dataset
ds_name = "voca";
% folder_name = "../../database/" + ds_name + "/" + ds_name + "-audio-cough";
folder_name = "../../../DataBase/" + ds_name + "/" + "audioDataFolder";
ads_voca = audioDatastore(folder_name, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

% load coughvid dataset
ds_name = "coughvid";
% folder_name = "../../database/" + ds_name + "/" + ds_name + "-audio-cough";
folder_name = "../../../DataBase/" + ds_name + "/" + "audioDataFolder";
ads_coughvid = audioDatastore(folder_name, ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

% choose if to shuffle the datasets
switch shuffle_dataset
    
    case "no"
        
        % do nothing. don't shuffle
        
    case "yes" % shuffle
        
        ads_voca = shuffle(ads_voca); % shuffle voca dataset
        ads_coughvid = shuffle(ads_coughvid); % shuffle coughvid dataset
        
end

% split voca dataset to (train, val, test)
[adsTrain_voca, adsVal_voca, adsTest_voca] = ...
    splitEachLabel(ads_voca, train_prcnt, val_prcnt, test_prcnt);

% split coughvid dataset to (train, val, test)
[adsTrain_coughvid, adsVal_coughvid, adsTest_coughvid] = ...
    splitEachLabel(ads_coughvid, train_prcnt, val_prcnt, test_prcnt);

% combine datasets
adsTrain = combine(adsTrain_voca, adsTrain_coughvid);
adsVal = combine(adsVal_voca, adsVal_coughvid);
adsTest = combine(adsTest_voca, adsTest_coughvid);

% adsTrain = {adsTrain_voca ; adsTrain_coughvid};
% adsVal = {adsVal_voca ; adsVal_coughvid};
% adsTest = {adsTest_voca ; adsTest_coughvid};

% % add labels to datasets
% adsTrain(:, 2) = {'voca' ; 'coughvid'};
% adsVal(:, 2) = {'voca' ; 'coughvid'};
% adsTest(:, 2) = {'voca' ; 'coughvid'};

% choose which dataset to use
switch dataset_name
    
    case "all"
        
        % do nothing. use all datasets
        
    case  "voca" % use only voca dataset
        
        adsTrain    =   adsTrain(1 , :);
        adsVal      =   adsVal(1 , :);
        adsTest     =   adsTest(1 , :);
        
    case "coughvid" % use only coughvid dataset
        
        adsTrain    =   adsTrain(2 , :);
        adsVal      =   adsVal(2 , :);
        adsTest     =   adsTest(2 , :);
        
end

end

