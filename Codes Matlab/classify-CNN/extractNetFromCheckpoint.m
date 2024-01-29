function net_classification_CRNN = extractNetFromCheckpoint(net_info, checkpointPath)

% find epoch with minimum loss on validation set
ValidationLosses = net_info.ValidationLoss; % load all losses
ValidationLosses(isnan(ValidationLosses)) = []; % remove nans
[~, min_ValidationLoss_indx] = min(ValidationLosses); % find indx with minimum value

% val error indx is related to net checkpoint at indx-1
min_ValidationLoss_indx = max(min_ValidationLoss_indx - 1, 1); % if indx = 0 -> indx = 1

% display epoch indx with minimum val loss
disp('best epoch:');
disp(num2str(min_ValidationLoss_indx));

% choose net from checkpoint folder at found epoch
nets_list = dir('checkpoint/'); % load metadata of all networks in checkpoint folder
nets_list([1 2],:) = []; % remove non relevent files

% nets are not in order by date -> sort by date
% get date numbers of all nets
datenums = zeros(length(nets_list), 1);
for i = 1 : length(datenums)
    datenums(i) = nets_list(i).datenum;
end

% sort dates
[~, datenums_indices] = sort(datenums);

% find net's name of best net (minimum loss on validation set)
net_crnn_name = nets_list(datenums_indices(min_ValidationLoss_indx)).name;

% load best net
net_classification_CRNN = load(checkpointPath + '/' + net_crnn_name);
net_classification_CRNN = net_classification_CRNN.net;

end

