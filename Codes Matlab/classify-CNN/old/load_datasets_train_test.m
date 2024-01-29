function [adsTrain, adsTest] = load_datasets_train_test(shuffle_dataset)

% percentages of (train, val, test) for splitEachLabel
train_prcnt = 0.8;
test_prcnt = 1 - train_prcnt;

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

% split voca dataset to (train, test)
[adsTrain_voca, adsTest_voca] = ...
    splitEachLabel(ads_voca, train_prcnt, test_prcnt);

% split coughvid dataset to (train, val, test)
[adsTrain_coughvid, adsTest_coughvid] = ...
    splitEachLabel(ads_coughvid, train_prcnt, test_prcnt);

adsTrain = combine(adsTrain_voca, adsTrain_coughvid);
adsTest = combine(adsTest_voca, adsTest_coughvid);

end
