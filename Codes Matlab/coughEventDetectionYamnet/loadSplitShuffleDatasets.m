function [adsTrain, adsVal, adsTest, adsSpecsTable] = ...
    loadSplitShuffleDatasets(datasetNames, splitPrcnts, shuffleDatasets)

% percentages of (train, val, test) for splitEachLabel
TrainValPrcnt = splitPrcnts(1) + splitPrcnts(2);
TrainPrcnt = splitPrcnts(1) / TrainValPrcnt;

% load ads (voca + coughvid + coswara)
ads = loadAds;

% remove datatsets not specified in datasetNames
ads = selectAdsBasedOnDsName(ads, datasetNames);

% split ads to train val test and add shuffle based on shuffleDatasets ("all" | "trainVal" | "no")
switch shuffleDatasets
    
    case "all"
        
        % shuffle ads => split ads to train&val / test => split train&val to train / val
        ads = shuffleAds(ads); % shuffle all ads
        [adsTrainVal, adsTest] = splitAds(ads, TrainValPrcnt); % split ads to train&val / test
        [adsTrain, adsVal] = splitAds(adsTrainVal, TrainPrcnt); % split adsTrainVal to train / val
        
    case "trainVal"
        
        % split ads to train&val / test => shuffle train&val => split train&val to train / val
        [adsTrainVal, adsTest] = splitAds(ads, TrainValPrcnt); % split ads to train&val / test
        adsTrainVal = shuffleAds(adsTrainVal); % shuffle train&val
        [adsTrain, adsVal] = splitAds(adsTrainVal, TrainPrcnt); % split adsTrainVal to train / val
        
    case "no"
        
        % split ads to train&val / test => split train&val to train / val
        [adsTrainVal, adsTest] = splitAds(ads, TrainValPrcnt); % split ads to train&val / test
        [adsTrain, adsVal] = splitAds(adsTrainVal, TrainPrcnt); % split adsTrainVal to train / val
        
    otherwise
        
        % display error in shuffleDatasets
        disp('Error in shuffleDatasets..');
end

% save specs in table
adsSpecsTable = extractSpecs(ads, adsTrain, adsVal, adsTest);

% sub-functions:

% ads = loadAds
% ads = selectAdsBasedOnDsName(ads, datasetNames)
% [adsA, adsB] = splitAds(ads, adsAPrcnt)
% ads = shuffleAds(ads)

% load all metadata files from voca, coughvid and coswara datasets and put
% all info in 1 ads file
    function ads = loadAds
        
        % set folders names
        folderNameVoca = "\\zigel-server-5\corona\DataBase\auto-segmentation-DL\voca\";
        folderNameCoughvid = "\\zigel-server-5\corona\DataBase\auto-segmentation-DL\coughvid\";
        
        % get metadata
        metadataVoca = struct2table(dir(folderNameVoca));
        metadataVoca = metadataVoca(3 : end, 1 : 2);
        metadataVoca(:, 1) = erase(metadataVoca{:, 1}, '.wav');
        
        metadataCoughvid = struct2table(dir(folderNameCoughvid));
        metadataCoughvid = metadataCoughvid(3 : end, 1 : 2);
        metadataCoughvid(:, 1) = erase(metadataCoughvid{:, 1}, '.wav');
        
        metadataCoswara = readtable("metadataCoswaraForCoughDetection.xlsx", ...
            'ReadVariableNames', true, ...
            'VariableNamingRule', 'modify');
        
        % create new table for all datasets
        ads = table;
        
        % fill table
        
        % set number per ds
        ads.dsNum = [ ...
            ones(height(metadataVoca), 1) ; ...
            ones(height(metadataCoughvid), 1) * 2 ; ...
            ones(height(metadataCoswara), 1) * 3 ; ...
            ];
        
        % add id
        % loop over all datasets and calsses
        for dsNum = 1 : max(ads.dsNum)
            
            switch dsNum
                case 1 % metadataVoca
                    ads.subjectId(ads.dsNum == 1) = metadataVoca.name;
                case 2 % metadataCoughvid
                    ads.subjectId(ads.dsNum == 2) = metadataCoughvid.name;
                case 3 % metadataCoswara
                    ads.subjectId(ads.dsNum == 3) = metadataCoswara.id;
            end
        end
        
        % add subject num
        % similar to id but now coswara also recieve numbers insted of name
        % loop over all datasets and calsses
        for dsNum = 1 : max(ads.dsNum)
            ads.subjectNum(ads.dsNum == dsNum) = num2cell((1 : length(ads.dsNum(ads.dsNum == dsNum)))');
        end
        
        % add audio file/folder location
        ads.location = [ ...
            metadataVoca.folder + "\" + metadataVoca.name; ...
            metadataCoughvid.folder + "\" + metadataCoughvid.name; ...
            string(metadataCoswara.folder) + metadataCoswara.id ...
            ];
        
        % add dataset
        ads.dataset = [ ...
            repmat("voca", height(metadataVoca), 1) ; ...
            repmat("coughvid", height(metadataCoughvid), 1) ; ...
            repmat("coswara", height(metadataCoswara), 1) ...
            ];
        
        % add nadav annotations to coswara dataset
        % ["N_Bd" "N_Bsh" "N_Ch" "N_Csh"]
        ads.N_Bd    = repmat("NA", height(ads), 1);
        ads.N_Bsh   = repmat("NA", height(ads), 1);
        ads.N_Ch    = repmat("NA", height(ads), 1);
        ads.N_Csh   = repmat("NA", height(ads), 1);
        ads.N_Bd(ads.dsNum == 3)    = metadataCoswara.N_Bd;
        ads.N_Bsh(ads.dsNum == 3)   = metadataCoswara.N_Bsh;
        ads.N_Ch(ads.dsNum == 3)    = metadataCoswara.N_Ch;
        ads.N_Csh(ads.dsNum == 3)   = metadataCoswara.N_Csh;
        
    end

% keep only datasets specified in datasetName and remove all others
    function ads = selectAdsBasedOnDsName(ads, datasetNames)
        
        % get names of datasets to remove
        dsNamesToRemove = ["voca" "coughvid" "coswara"]; % names of all datasets
        dsNamesToRemove = setxor(dsNamesToRemove, datasetNames); % names to remove
        
        % % remove datasets not specified in datasetName
        ads(contains(ads.dataset,dsNamesToRemove), :) = [];
        
        % update dsClassNum based on remaining datasets:
        
        % indices of change in ds/class
        diffDsNum = [ads.dsNum(1) ; diff(ads.dsNum)];
        DsNumChange = [find(diffDsNum ~= 0) ; length(diffDsNum)];
        
        % loop over all ds/classes => set dsClassNum to be consecutive
        % (example: [44, 5, 888] => [11, 2, 333])
        for row = 1 : length(DsNumChange) - 1
            indicesToChange = DsNumChange(row) : DsNumChange(row + 1) - 1;
            ads.dsNum(indicesToChange) = row;
        end
    end

% split ads to 2 adsets: train&val and test
    function [adsA, adsB] = splitAds(ads, adsAPrcnt)
        
        % pre-allocation
        adsA = ads;
        adsB = ads;
        adsAIndices = [];
        adsBIndices = [];
        
        % loop over all datasets and calsses
        for dsNum = min(ads.dsNum) : max(ads.dsNum)
            
            FirstIndx = find(ads.dsNum == dsNum, 1, 'first'); % 1st indx
            LastIndx = find(ads.dsNum == dsNum, 1, 'last'); % 2nd indx
            adsALen = round((LastIndx - FirstIndx + 1) * adsAPrcnt); % length of adsA
            adsAIndices = cat(2, adsAIndices, FirstIndx : FirstIndx + adsALen - 1); % indices of adsA
            adsBIndices = cat(2, adsBIndices, FirstIndx + adsALen : LastIndx); % indices of adsB
        end
        
        % remove indices from ads
        adsA(adsBIndices, :) = []; % remove B indices from adsA
        adsB(adsAIndices, :) = []; % remove A indices from adsB
        
    end

% shuffle ads separately per each dataset/class
    function ads = shuffleAds(ads)
        
        % loop over all datasets and calsses
        for dsNum = min(ads.dsNum) : max(ads.dsNum)
            
            FirstIndx = find(ads.dsNum == dsNum, 1, 'first'); % 1st indx
            LastIndx = find(ads.dsNum == dsNum, 1, 'last'); % 2nd indx
            indiceslen = LastIndx - FirstIndx + 1; % length of interval
            randIndices = randperm(indiceslen) + FirstIndx - 1; % create random order for indices
            
            % change indices order
            ads(ads.dsNum == dsNum, :) = ads(randIndices, :);
        end
        
    end

    function adsSpecsTable = extractSpecs(ads, adsTrain, adsVal, adsTest)
        % pre-allocation
        adsCell = {ads adsTrain adsVal adsTest};
        
        % create table for scores
        VariableTypesAndNames = [...
            "string"	"all sets"
            "string"	"train"
            "string"	"validation"
            "string"	"test"
            ];
        RowNames = {'No. subjects' ...
            'No. subjects voca' 'No. subjects coughvid' 'No. subjects coswara' ...
            'No. records' ...
            'No. records voca' 'No. records coughvid' 'No. records coswara' ...
            };
        adsSpecsTable = table( ...
            'size', [length(RowNames), height(VariableTypesAndNames)], ...
            'VariableTypes', VariableTypesAndNames(:, 1), ...
            'VariableNames', VariableTypesAndNames(:, 2), ...
            'RowNames', RowNames ...
            );
        
        % loop over all sets
        for setIndx = 1 : length(adsCell)
            
            ads = adsCell{setIndx};
            adsMultiRec = addMultiRecToAds(ads);
            
            % No. subjects
            numSubj = height(ads);
            adsSpecsTable(1, setIndx) = {string(numSubj)};
            
            % No. subjects voca coughvid coswara
            adsSpecsTable(2, setIndx) = {string(height(ads(ads.dataset == "voca", :)))};
            adsSpecsTable(3, setIndx) = {string(height(ads(ads.dataset == "coughvid", :)))};
            adsSpecsTable(4, setIndx) = {string(height(ads(ads.dataset == "coswara", :)))};
            
            % No. recordings
            numRec = height(adsMultiRec);
            adsSpecsTable(5, setIndx) = {numRec};
            
            % No. recordings voca coughvid coswara
            adsSpecsTable(6, setIndx) = {string(height(adsMultiRec(adsMultiRec.dataset == "voca", :)))};
            adsSpecsTable(7, setIndx) = {string(height(adsMultiRec(adsMultiRec.dataset == "coughvid", :)))};
            adsSpecsTable(8, setIndx) = {string(height(adsMultiRec(adsMultiRec.dataset == "coswara", :)))};
        end
    end
    
    % change ads with respect to multipal recordings per subject
    % add rows for each extra recording
    function adsMultiRec = addMultiRecToAds(ads)
        
        % initialization
        MultiRecIndx = 0;
        
        % loop over all rows of ads (all subjects)
        for row = 1 : height(ads)
            
            MultiRecIndx = MultiRecIndx + 1;
            adsMultiRec(MultiRecIndx, :) = ads(row, :); % update adsMultiRec
            
            % check if there is extra recording
            if  ads.N_Ch(row) == "Good" && ads.N_Csh(row) == "Good"
                
                MultiRecIndx = MultiRecIndx + 1;
                adsMultiRec(MultiRecIndx, :) = ads(row, :); % update adsMultiRec
            end
        end
        
    end

end

