function [Spectrograms, adsSpecs] = preprocessSegYamnet(ads, winInfo, afe, coughEventDetect)

% pre-allocation

% No. spectrums (1xn vector) per spectrogram (mxn matrix)
numSpectrumsPerSpectrogram = winInfo.numSpectrumsPerSpectrogram;

% No. spectrums that not overlap between each 2 spectrograms
numSpectrumsHopBetweenSpectrograms = winInfo.numSpectrumsHopBetweenSpectrograms;

Spectrograms = []; % mel spectrograms
audioStartIndx = 1;
audioTypes = ["cough-heavy" "cough-shallow"]; % types of coughs in coswara dataset
ourAnnotations = [ads.N_Ch ads.N_Csh];

% info on audio signals in Features matrix to identify labels original
% locations after prediction
VariableTypesAndNames = [...
    "string"        "dataset"
    "double"        "subjectNum"
    "string"        "audioType"
    "categorical" 	"class"
    "double"        "age"
    "string"        "gender"
    "string"        "symptoms"
    "double"        "audioStartIndx"
    "double"        "audioEndIndx"
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
        
        if ads.dataset{row} == "coswara" % coswara
            
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
            fileLocation = ads.location(row);
            [audioSignal, ~] = audioread(fileLocation);
        end
        
        spectrogram = extract(afe, audioSignal); % extract mel spectrograms
        spectrogram = log10(spectrogram + single(0.001)); % ln(result + epsilon)
        
        % choose if to use cough event detection or not
        if coughEventDetect == "no"
            
            % basic threshold to remove parts at begining and end of signal
            % with low energy
            [N,edges] = histcounts(spectrogram);
            edges_diff = edges(2) - edges(1);
            threshold = edges(N == max(N)) + edges_diff;
            
            % remove bad rows
            startEnergyIndx = find(mean(spectrogram, 2) >= threshold, 1);
            EndEnergyIndx = find(mean(spectrogram, 2) >= threshold, 1, 'last');
            spectrogram = spectrogram(startEnergyIndx : EndEnergyIndx, :);
            
            % if spectrogram is empty => dont add to ds
            if height(spectrogram) < 96
                continue;
            end
        end
        
        [numSpectrums, ~] = size(spectrogram); % No. spectrums (1xn vectors)
        
        % No. spectrograms (mxn matrices)
        numSpectrograms = ...
            floor((numSpectrums - numSpectrumsPerSpectrogram)/numSpectrumsHopBetweenSpectrograms) + 1;
        
        % add all spectrograms of signal to collection
        for hop = 1 : numSpectrograms
            range = 1 + numSpectrumsHopBetweenSpectrograms*(hop-1):numSpectrumsHopBetweenSpectrograms*(hop-1) + numSpectrumsPerSpectrogram;
            Spectrograms = cat(4, Spectrograms, spectrogram(range,:));
        end
        
        % save signals specification
        audioEndIndx = audioStartIndx + numSpectrograms - 1;
        
        % ads_specs = [data source, No. signal, start index, end index]
        adsSpecs = cat(1, adsSpecs, {ads.dataset{row}, ads.subjectNum{row}, audioType, ads.class{row}, ads.age(row), ads.gender{row}, ads.symptoms{row}, audioStartIndx, audioEndIndx});
        audioStartIndx = audioEndIndx + 1;
    end
end

end

