function newAds = loadNewAds(folderName)

metadataNewAds = struct2table(dir(folderName));
metadataNewAds = metadataNewAds(3 : end, 1 : 2);
metadataNewAds(:, 1) = erase(metadataNewAds{:, 1}, '.wav');

newAds = table;

% fill table

% set number per ds
newAds.dsNum = ones(height(metadataNewAds), 1);

% add id
newAds.subjectId(newAds.dsNum == 1) = metadataNewAds.name;

% add subject num
newAds.subjectNum = num2cell((1 : length(newAds.dsNum))');

% add audio file/folder location
newAds.location = metadataNewAds.folder + "\" + metadataNewAds.name;

% add dataset
newAds.dataset = repmat("newAds", height(metadataNewAds), 1);

% add nadav annotations to coswara dataset
% ["N_Bd" "N_Bsh" "N_Ch" "N_Csh"]
newAds.N_Bd    = repmat("NA", height(newAds), 1);
newAds.N_Bsh   = repmat("NA", height(newAds), 1);
newAds.N_Ch    = repmat("NA", height(newAds), 1);
newAds.N_Csh   = repmat("NA", height(newAds), 1);

end

