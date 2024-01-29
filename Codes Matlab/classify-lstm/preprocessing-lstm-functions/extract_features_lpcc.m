function [Features, trueLabels, adsSpecs] = extract_features_lpcc(ADS, coughEventLocations)

% pre-allocation
win_info = extrct_win_info();
Features = []; % LPCC
trueLabels = []; % true labels (positive/negative)
indx_start_sig = 1;
indx_audioIn = 0;
p = 12;
% num_LPCC = 12;
num_coeffs = 12;

% info on audio signals in Features matrix to identify labels original
% locations after prediction
adsSpecs = table( ...
    'size', [0, 5], ...
    'VariableTypes', {'string' 'double' 'categorical' 'double' 'double'}, ...
    'VariableNames',{'database' 'sig_num' 'label' 'start_indx' 'end_indx'} ...
    );

% loop over all audio file in audio set

[num_datasets, ~] = size(ADS);

for i = 1 : num_datasets
    
    ads = ADS{i, 1}; % audioDataStore in cell
    
    reset(ads); % so hasdata will start from beginning
    while hasdata(ads)
        
        [audioIn, fileInfo] = read(ads); % load audio signal from ads
        indx_audioIn = indx_audioIn + 1;
        num_cough_events = length(coughEventLocations{indx_audioIn, 1});
        
        for cough_event = 1 : num_cough_events
            
            start_indx_cough_event = coughEventLocations{indx_audioIn, 1}(cough_event);
            end_indx_cough_event = coughEventLocations{indx_audioIn, 2}(cough_event);
            
            s_t = audioIn(start_indx_cough_event : end_indx_cough_event, 1);
            
            % buffer
            S_t = buffer(s_t, ...
                win_info.win_len, ...
                win_info.numSamplesOverlapBetweenWin, ...
                'nodelay');
            
            % LPC
            [lpc_coeffs, predErrorVar] = lpc(S_t, p);
            lpc_coeffs = lpc_coeffs';
            %             lpc_coeffs = lpc_coeffs(2:end, :);
            
            %             % convert NaN values to zeros
            %             lpc_coeffs(isnan(lpc_coeffs)) = 0;
            
            % convert lpc coefficients to cepstral coefficients
            lpcc = lpc2lpcc(lpc_coeffs, predErrorVar);
            
            % convert NaN values to zeros
            lpcc(isnan(lpcc)) = 0;
            
            %             % select what features to extract from each audio signal
            %             afe = audioFeatureExtractor(...
            %                 'SampleRate',       16000, ...
            %                 'Window',           hann(256, 'periodic'), ...
            %                 'OverlapLength',    200, ...
            %                 'FFTLength',        512, ...
            %                 'mfcc',             true ...
            %                 );
            %             setExtractorParams( ...
            %                 afe,            'mfcc', ...
            %                 'NumCoeffs',    num_coeffs ...
            %                 );
            
            %             Features = cat(1, Features, {extract(afe, s_t)'});
            Features = cat(1, Features, {lpcc});
            trueLabels = cat(1, trueLabels, fileInfo.Label);
        end
        
        % save signals specification
        [~, name, ~] = fileparts(fileInfo.FileName);
        indx_stop_sig = indx_start_sig + num_cough_events - 1;
        adsSpecs = cat(1, adsSpecs, {string(ADS{i,2}), str2double(name), fileInfo.Label, indx_start_sig, indx_stop_sig});
        indx_start_sig = indx_stop_sig + 1;
    end
end

end

