function [adsTrain, adsVal, adsTest, adsSpecsTable] = ...
    loadSplitShuffleDatasets(datasetNames, splitPrcnts, shuffleDatasets, changeSubjectsOrder)

% percentages of (train, val, test) for splitEachLabel
TrainValPrcnt = splitPrcnts(1) + splitPrcnts(2);
TrainPrcnt = splitPrcnts(1) / TrainValPrcnt;

% load ads (voca + coughvid + coswara)
ads = loadAds;

% remove datatsets not specified in datasetNames
ads = selectAdsBasedOnDsName(ads, datasetNames);

% change order of subjects to be same as old method
if changeSubjectsOrder == "yes"
    ads = changeOrder2Compare2OldMethod(ads);
end

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
% ads = changeOrder2Compare2OldMethod(ads)
% [adsA, adsB] = splitAds(ads, adsAPrcnt)
% ads = shuffleAds(ads)

% load all metadata files from voca, coughvid and coswara datasets and put
% all info in 1 ads file
    function ads = loadAds
        
        % set folder path
        prev_folder = cd('..\');
        folder_path = pwd;
        folder_name = folder_path + "\metadata\";
        cd(prev_folder);
        
        % load metadata files
        metadataVocaNeg     = readtable(folder_name + "metadata_voca_negative.xlsx");
        metadataVocaPos     = readtable(folder_name + "metadata_voca_positive.xlsx");
        metadataCoughvid    = readtable(folder_name + "metaDataCoughvid.xlsx");
        metadataCoswara     = readtable(folder_name + "metadata coswara final.csv");
        
        % split coswara to neg and pos
        metadataCoughvidNeg = metadataCoughvid(string(metadataCoughvid.class) == "negative", :);
        metadataCoughvidPos = metadataCoughvid(string(metadataCoughvid.class) == "positive", :);
        
        % split coswara to neg and pos
        metadataCoswaraNeg = metadataCoswara(string(metadataCoswara.class) == "negative", :);
        metadataCoswaraPos = metadataCoswara(string(metadataCoswara.class) == "positive", :);
        
        % create new table for all datasets
        ads = table;
        
        % fill table
        
        % set number per ds and class
        ads.dsClassNum = [ ...
            ones(height(metadataVocaNeg), 1) ; ...
            ones(height(metadataVocaPos), 1) * 2 ; ...
            ones(height(metadataCoughvidNeg), 1) * 3 ; ...
            ones(height(metadataCoughvidPos), 1) * 4 ; ...
            ones(height(metadataCoswaraNeg), 1) * 5 ; ...
            ones(height(metadataCoswaraPos), 1) * 6 ...
            ];
        
        % add id
        % loop over all datasets and calsses
        for dsClassNum = 1 : max(ads.dsClassNum)
            
            switch dsClassNum
                case 3 % metadataCoughvidNeg
                    ads.id(ads.dsClassNum == 3) = metadataCoughvidNeg.uuid;
                case 4 % metadataCoughvidPos
                    ads.id(ads.dsClassNum == 4) = metadataCoughvidPos.uuid;
                case 5 % metadataCoswaraNeg
                    ads.id(ads.dsClassNum == 5) = metadataCoswaraNeg.id;
                case 6 % metadataCoswaraPos
                    ads.id(ads.dsClassNum == 6) = metadataCoswaraPos.id;
                otherwise % all others (set numbers)
                    ads.id(ads.dsClassNum == dsClassNum) = num2cell((1 : length(ads.dsClassNum(ads.dsClassNum == dsClassNum)))');
            end
        end
        
        % add subject num
        % similar to id but now coswara also recieve numbers insted of name
        % loop over all datasets and calsses
        for dsClassNum = 1 : max(ads.dsClassNum)
            ads.subjectNum(ads.dsClassNum == dsClassNum) = num2cell((1 : length(ads.dsClassNum(ads.dsClassNum == dsClassNum)))');
        end
        
        % add audio file/folder location
        ads.location = [ ...
            metadataVocaNeg.audio_file_location ; ...
            metadataVocaPos.audio_file_location ; ...
            metadataCoughvidNeg.audioFileLocation ; ...
            metadataCoughvidPos.audioFileLocation ; ...
            string(metadataCoswaraNeg.folder) + string(metadataCoswaraNeg.id) ; ...
            string(metadataCoswaraPos.folder) + string(metadataCoswaraPos.id) ...
            ];
        
        % add dataset
        ads.dataset = [ ...
            metadataVocaNeg.ds ; ...
            metadataVocaPos.ds ; ...
            metadataCoughvidNeg.dataset ; ...
            metadataCoughvidPos.dataset ; ...
            metadataCoswaraNeg.dataset ; ...
            metadataCoswaraPos.dataset ...
            ];
        
        % add class
        ads.class = [ ...
            metadataVocaNeg.class ; ...
            metadataVocaPos.class ; ...
            metadataCoughvidNeg.class ; ...
            metadataCoughvidPos.class ; ...
            metadataCoswaraNeg.class ; ...
            metadataCoswaraPos.class ...
            ];
        
        % add age
        ads.age = [ ...
            metadataVocaNeg.age ; ...
            metadataVocaPos.age ; ...
            metadataCoughvidNeg.age ; ...
            metadataCoughvidPos.age ; ...
            metadataCoswaraNeg.a ; ...
            metadataCoswaraPos.a ...
            ];
        
        % add gender
        ads.gender = [ ...
            metadataVocaNeg.gender ; ...
            metadataVocaPos.gender ; ...
            metadataCoughvidNeg.gender ; ...
            metadataCoughvidPos.gender ; ...
            metadataCoswaraNeg.g ; ...
            metadataCoswaraPos.g ...
            ];
        
        % add symptoms (yes/no only for coswara)
        ads.symptoms = repmat("NA", height(ads), 1);
        ads.symptoms(ads.dsClassNum == 5 | ads.dsClassNum == 6) = [metadataCoswaraNeg.symptoms ; metadataCoswaraPos.symptoms];
        ads.symptoms(ads.symptoms == '') = "NA";
        
        % add nadav annotations to coswara dataset
        % ["N_Bd" "N_Bsh" "N_Ch" "N_Csh"]
        ads.N_Bd = repmat("NA", height(ads), 1);
        ads.N_Bsh = repmat("NA", height(ads), 1);
        ads.N_Ch = repmat("NA", height(ads), 1);
        ads.N_Csh = repmat("NA", height(ads), 1);
        ads.N_Bd(ads.dsClassNum == 5 | ads.dsClassNum == 6) = [metadataCoswaraNeg.N_Bd ; metadataCoswaraPos.N_Bd];
        ads.N_Bsh(ads.dsClassNum == 5 | ads.dsClassNum == 6) = [metadataCoswaraNeg.N_Bsh ; metadataCoswaraPos.N_Bsh];
        ads.N_Ch(ads.dsClassNum == 5 | ads.dsClassNum == 6) = [metadataCoswaraNeg.N_Ch ; metadataCoswaraPos.N_Ch];
        ads.N_Csh(ads.dsClassNum == 5 | ads.dsClassNum == 6) = [metadataCoswaraNeg.N_Csh ; metadataCoswaraPos.N_Csh];
        
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
        diffDsClassNum = [ads.dsClassNum(1) ; diff(ads.dsClassNum)];
        DsClassNumChange = [find(diffDsClassNum ~= 0) ; length(diffDsClassNum)];
        
        % loop over all ds/classes => set dsClassNum to be consecutive
        % (example: [44, 5, 888] => [11, 2, 333])
        for row = 1 : length(DsClassNumChange) - 1
            indicesToChange = DsClassNumChange(row) : DsClassNumChange(row + 1) - 1;
            ads.dsClassNum(indicesToChange) = row;
        end
    end

% change order of subjects to be same as old method
    function ads = changeOrder2Compare2OldMethod(ads)
        
        % loop over all datasets and calsses
        for dsClassNum = 1 : max(ads.dsClassNum)
            ads.subjectNum(ads.dsClassNum == dsClassNum) = ...
                num2cell(double(sort(string(ads.subjectNum(ads.dsClassNum == dsClassNum)))));
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
        for dsClassNum = min(ads.dsClassNum) : max(ads.dsClassNum)
            
            FirstIndx = find(ads.dsClassNum == dsClassNum, 1, 'first'); % 1st indx
            LastIndx = find(ads.dsClassNum == dsClassNum, 1, 'last'); % 2nd indx
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
        for dsClassNum = min(ads.dsClassNum) : max(ads.dsClassNum)
            
            FirstIndx = find(ads.dsClassNum == dsClassNum, 1, 'first'); % 1st indx
            LastIndx = find(ads.dsClassNum == dsClassNum, 1, 'last'); % 2nd indx
            indiceslen = LastIndx - FirstIndx + 1; % length of interval
            randIndices = randperm(indiceslen) + FirstIndx - 1; % create random order for indices
            
            % change indices order
            ads(ads.dsClassNum == dsClassNum, :) = ads(randIndices, :);
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
        RowNames = {'No. subjects' 'No. Subjects (positive class)' 'No. subjects (negative class)' ...
            'No. recordings' 'No. recordings (positive class)' 'No. recordings (negative class)' ...
            'age' ...
            'No. male subjects' 'No. female subjects' ...
            'No. male subjects recordings' 'No. female subjects recordings' ...
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
            
            % No. subjects (positive class)
            numSubjPos = ...
                string(sum(ads.class == "positive")) + ' (' + ...
                string(round(sum(ads.class == "positive") / numSubj  * 100, 1)) + ...
                '%)';
            adsSpecsTable(2, setIndx) = {numSubjPos};
            
            % No. subjects (negative class)
            numSubjNeg = ...
                string(sum(ads.class == "negative")) + ' (' + ...
                string(round(sum(ads.class == "negative") / numSubj  * 100, 1)) + ...
                '%)';
            adsSpecsTable(3, setIndx) = {numSubjNeg};
            
            % No. recordings
            numRec = height(adsMultiRec);
            adsSpecsTable(4, setIndx) = {numRec};
            
            % No. recordings (positive class)
            numRecPos = ...
                string(sum(adsMultiRec.class == "positive")) + ' (' + ...
                string(round(sum(adsMultiRec.class == "positive") / numRec  * 100, 1)) + ...
                '%)';
            adsSpecsTable(5, setIndx) = {numRecPos};
            
            % No. recordings (negative class)
            numRecNeg = ...
                string(sum(adsMultiRec.class == "negative")) + ' (' + ...
                string(round(sum(adsMultiRec.class == "negative") / numRec  * 100, 1)) + ...
                '%)';
            adsSpecsTable(6, setIndx) = {numRecNeg};
            
            % age mean & std
            ageMeanStd = ...
                string(round(mean(ads.age, 'omitnan'), 1)) + ' ' + char(177) + ' ' + ...
                string(round(std(ads.age, 'omitnan'), 1));
            adsSpecsTable(7, setIndx) = {ageMeanStd};
            
            % No. male subjects
            numMale = ...
                string(sum(strcmpi(ads.gender, "male"))) + ' (' + ...
                string(round(sum(strcmpi(ads.gender, "male")) / numSubj  * 100, 1)) + '%)';
            adsSpecsTable(8, setIndx) = {numMale};
            
            % No. female subjects
            numFemale = ...
                string(sum(strcmpi(ads.gender, "female"))) + ' (' + ...
                string(round(sum(strcmpi(ads.gender, "female")) / numSubj  * 100, 1)) + '%)';
            adsSpecsTable(9, setIndx) = {numFemale};
            
            % No. male subjects recordings
            numMale = ...
                string(sum(strcmpi(adsMultiRec.gender, "male"))) + ' (' + ...
                string(round(sum(strcmpi(adsMultiRec.gender, "male")) / numRec  * 100, 1)) + '%)';
            adsSpecsTable(10, setIndx) = {numMale};
            
            % No. female subjects recordings
            numFemale = ...
                string(sum(strcmpi(adsMultiRec.gender, "female"))) + ' (' + ...
                string(round(sum(strcmpi(adsMultiRec.gender, "female")) / numRec  * 100, 1)) + '%)';
            adsSpecsTable(11, setIndx) = {numFemale};
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

