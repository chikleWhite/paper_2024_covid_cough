function [Spectrograms, lblsTrueDec, adsSpecs] = preprocessSegYamnet(ads, afe, winInfo, addTrueLbls, remLowMagSeg)

% No. spectrums (1xn vector) per spectrogram (mxn matrix)
numSpectrumsPerSpectrogram = winInfo.numSpectrumsPerSpectrogram;

% No. spectrums that not overlap between each 2 spectrograms
numSpectrumsHopBetweenSpectrograms = winInfo.numSpectrumsHopBetweenSpectrograms;

Spectrograms = []; % mel spectrograms
lblsTrueDec = []; % true lables (per segment)
audioStartIndx = 1;
audioTypes = ["cough-heavy" "cough-shallow"]; % types of coughs in coswara dataset
ourAnnotations = [ads.N_Ch ads.N_Csh];

% info on audio signals in Features matrix to identify labels original
% locations after prediction
VariableTypesAndNames = [...
    "string"        "dataset"
    "double"        "subjectId"
    "double"        "subjectNum"
    "string"        "audioType"
    "double"        "audioStartIndx"
    "double"        "audioEndIndx"
    "double"        "audioTailIndx"
    ];
adsSpecs = table( ...
    'size', [0, height(VariableTypesAndNames)], ...
    'VariableTypes', VariableTypesAndNames(:, 1), ...
    'VariableNames', VariableTypesAndNames(:, 2) ...
    );

% loop over all audio file in ads

for row = 1 : height(ads)
    
    % loop over audio types
    for audioIndx = 1 : length(audioTypes)
        
        % set loading method based on dataset:
        
        if ads.dataset(row) == "coswara" % coswara
            
            audioType = audioTypes(audioIndx); % name of audio type
            
            % check if audioType is good (Good | NoGood)
            % if not good => skip
            if ourAnnotations{row, audioIndx} == "NoGood"
                continue;
            end
            
            % load audio signal
            fileLocation = ads.location(row) + "\" + audioType + ".wav";
            [audioSignal, ~] = audioread(fileLocation);
            
        else % voca | coughvid
            
            % 2nd iteration => exit inner loop
            if audioIndx == 2
                break;
            end
            
            % 1st iteration => load audio signal
            
            audioType = "cough"; % name of audio type
            
            % load audio signal
            fileLocation = ads.location(row) + ".wav";
            [audioSignal, ~] = audioread(fileLocation);
        end
        
        spectrogram = extract(afe, audioSignal); % extract mel spectrograms
        spectrogram = log10(spectrogram + single(0.001)); % ln(result + epsilon)
        [numSpectrums, ~] = size(spectrogram); % No. spectrums (1xn vectors)
        
        % threshold for removal of low magnitude segments
        [N, edges] = histcounts(mean(spectrogram, 2));
        edgesDiff = edges(2) - edges(1);
        [~, maxIndx] = max(N);
        threshold = edges(maxIndx) + edgesDiff;
        
        % No. spectrograms (mxn matrices)
        numSpectrograms = ...
            floor((numSpectrums - numSpectrumsPerSpectrogram)/numSpectrumsHopBetweenSpectrograms) + 1;
        
        % find last indx of signal where there is integer number of spectrograms
        numSpectrumsMod = numSpectrumsPerSpectrogram + ...
            numSpectrumsHopBetweenSpectrograms * (numSpectrograms - 1);
        audioTailIndx = winInfo.winLen + ...
            winInfo.numSamplesHopBetweenWin * (numSpectrumsMod - 1);
        
        % choose if to add true labels
        switch addTrueLbls
            
            case "yes"
                
                % extract true labels (per segment)
                [~, lblsPerRecord] = loadSegTrueLblsYamnet(audioSignal, ads(row, :), audioType, numSpectrograms, audioTailIndx, winInfo);
                
                % pre-allocation for segments removal
                segRemIndices = zeros(size(lblsPerRecord)); % removal locations
                numSegRem = 0; % No. segments to remove
                
                % add all spectrograms of signal to collection
                for hop = 1 : numSpectrograms
                    range = ...
                        1 + numSpectrumsHopBetweenSpectrograms * (hop - 1) : ...
                        numSpectrumsHopBetweenSpectrograms * (hop - 1) + numSpectrumsPerSpectrogram;
                    
                    % check if segment is non-cough and below threshold
                    if ...
                            remLowMagSeg == "yes" && ... % removal is allowed by user (during model train)
                            lblsPerRecord(hop) == "nonCough" && ... % non-cough segment
                            mean(spectrogram(range, :), 'all') < threshold && ... % avg of spectrogram is below threshold
                            rand(1) >= 0.7 % set 50% chance for removing segment
                        
                        % don't add spectrogram to collection
                        % save removing indices
                        segRemIndices(hop) = 1;
                        numSegRem = numSegRem + 1;
                        
                    else
                        % add spectrogram to collection
                        Spectrograms = cat(4, Spectrograms, spectrogram(range,:));
                    end
                end
                
                segRemIndices = logical(segRemIndices); % turn to logical
                lblsPerRecord(segRemIndices) = []; % remove true labels of removed segment
                numSpectrograms = numSpectrograms - numSegRem; % update No. segments
                lblsTrueDec = cat(1, lblsTrueDec, lblsPerRecord); % add true labels to collection
                
            case "no"
                
                % don't extract true labels
                
                % add all spectrograms of signal to collection
                for hop = 1 : numSpectrograms
                    range = 1 + numSpectrumsHopBetweenSpectrograms * (hop - 1): ...
                        numSpectrumsHopBetweenSpectrograms * (hop - 1) + numSpectrumsPerSpectrogram;
                    Spectrograms = cat(4, Spectrograms, spectrogram(range,:));
                end
        end
        
        % save signals order and No. segments per signal
        audioEndIndx = audioStartIndx + numSpectrograms - 1;
        adsSpecs = cat(1, adsSpecs, {ads.dataset(row), ads.subjectId{row}, ads.subjectNum{row}, audioType, audioStartIndx, audioEndIndx, audioTailIndx});
        audioStartIndx = audioEndIndx + 1;
        
        %     % print progress done
        %     prcnt_done = round(progress(ads) * 100);
        %     fprintf([repmat('\b', 1, 5), ' %d%%'], prcnt_done);
    end
end

end

