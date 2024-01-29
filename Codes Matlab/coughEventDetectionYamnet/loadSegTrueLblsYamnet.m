function [lbls, lblsDec] = loadSegTrueLblsYamnet(audioSignal, ads, audioType, numSpectrograms, audioTailIndx, winInfo)

% pre-allocation
audioSignal = audioSignal(1 : audioTailIndx);
sigLen = length(audioSignal);
lbls = zeros(sigLen, 1);
lblsDec = zeros(numSpectrograms, 1);
fs = 16000; % 16 [kHz]
fsDec = fs * numSpectrograms / (sigLen - winInfo.segmentLen);

% get file path of true labels
annotationPath = ...
    "\\zigel-server-5\corona\Codes\Covid-19 diagnosis using cough and breathing audio signal processing\";
switch ads.dataset
    case "voca"
        annotationFileName = annotationPath + "voca\true seg labels\" + ads.subjectId + ".txt";
    case "coughvid"
        annotationFileName = annotationPath + "coughvid\true seg labels\" + ads.subjectId + ".txt";
    case "coswara"
        annotationFileName = annotationPath + "coswara\true seg labels\" + ads.subjectId + "\" + audioType + ".txt";
end

% load labels txt file
timeStamps = readcell(annotationFileName, 'Delimiter', '\t');

% run over all cough events ----------------------------------------------

for j = 1 : size(timeStamps, 1)
    
    % find cough event
%     if ~isempty(intersect(string(timeStamps{j,3}), ["cough", "1", "2", "3"]))
    if string(timeStamps{j,3}) == "cough"
        
        startTime = timeStamps{j,1};
        endTime = timeStamps{j,2};
        lbls(max(1, floor(startTime * fs)) : ceil(endTime * fs)) = 1;
        lblsDec(max(1, floor(startTime * fsDec)) : ceil(endTime * fsDec)) = 1;
        continue;
    end
    
    % find 1st phase
    if string(timeStamps{j,3}) == "1"
        
        startTime = timeStamps{j,1};
        continue;
    end
    
    % find 2nd phase
    if string(timeStamps{j,3}) == "2"
        
        endTime = timeStamps{j,2};
        lbls(floor(startTime * fs) : ceil(endTime * fs)) = 1;
        lblsDec(floor(startTime * fsDec) : ceil(endTime * fsDec)) = 1;
        continue;
    end
    
    % find 3rd phase
    if string(timeStamps{j,3}) == "3"
        
        endTime = timeStamps{j,2};
        lbls(floor(startTime * fs) : ceil(endTime * fs)) = 1;
        lblsDec(floor(startTime * fsDec) : ceil(endTime * fsDec)) = 1;
    end
end

lbls = lbls(1 : audioTailIndx);
lblsDec = lblsDec(1 : numSpectrograms);

lbls = categorical(lbls, [0 1], {'nonCough' 'Cough'});
lblsDec = categorical(lblsDec, [0 1], {'nonCough' 'Cough'});


% figure;
% 
% timeEnd = length(lbls) / fs;
% time = linspace(0, timeEnd, length(lbls));
% 
% % time_start_dec = win_info.segment_len / 2 / fs;
% % time_end_dec = (length(lbls) - win_info.segment_len/2)/ fs;
% % time_dec = linspace(time_start_dec, time_end_dec, length(lbls_dec));
% 
% timeEndDec = (length(lbls) - winInfo.segmentLen)/ fs;
% timeDec = linspace(0, timeEndDec, length(lblsDec));
% 
% subplot(3,1,1);
% plot(time, audioSignal);
% axis tight;
% xlim([0 timeEnd]);
% title('signal');
% 
% subplot(3,1,2);
% plot(time, lbls);
% axis tight;
% xlim([0 timeEnd]);
% title('labels');
% 
% subplot(3,1,3);
% plot(timeDec, lblsDec);
% axis tight;
% xlim([0 timeEnd]);
% title('labels after decimation');
% 
% sgtitle([ads.dataset, ' ', ads.subjectNum]);

end

