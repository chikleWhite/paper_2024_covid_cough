function winInfo = extrctWinInfo(winInfo)

% add previously selected info:

% add previous data
winLen = winInfo.winLen;
overlapCoeffWin = winInfo.overlapCoeffWin;
overlapCoeffSpectrograms = winInfo.overlapCoeffSpectrograms;

% spectrogram info
SpectrumType = winInfo.SpectrumType;
WindowNormalization = winInfo.WindowNormalization;
FilterBankNormalization = winInfo.FilterBankNormalization;

% set win info

fs = 16e3; % sampling frequency
win = hann(winLen, 'periodic'); % window type
numSamplesOverlapBetweenWin = round(winLen*overlapCoeffWin); % overlap length
numSamplesHopBetweenWin = winLen - numSamplesOverlapBetweenWin; % hop length
fftLenCoeff = 2; % fft length ratio coefficient
fftLen = winLen * fftLenCoeff; % fft length
numBands = 64; % y axis in spectrogram

% No. spectrums (nx1 vector) per spectrogram (nxm matrix)
numSpectrumsPerSpectrogram = 96;

% No. spectrums that overlap between each 2 spectrograms
numSpectrumsOverlapBetweenSpectrograms = ...
    round(numSpectrumsPerSpectrogram * overlapCoeffSpectrograms);

% No. spectrums that not overlap between each 2 spectrograms
numSpectrumsHopBetweenSpectrograms = ...
    numSpectrumsPerSpectrogram - numSpectrumsOverlapBetweenSpectrograms;

% No. samples per spectrogram (segment)
segmentLen = (numSpectrumsPerSpectrogram - 1) * numSamplesHopBetweenWin + winLen;

% No. samples that hop between each spectrogram (segment)
numSamplesHopBetweenSpectrograms = numSpectrumsHopBetweenSpectrograms * numSamplesHopBetweenWin;

% save in struct
winInfo = struct(...
    'fs', fs, ...
    'winLen', winLen, ...
    'win', win, ...
    'overlapCoeffWin', overlapCoeffWin, ...
    'numSamplesOverlapBetweenWin', numSamplesOverlapBetweenWin, ...
    'numSamplesHopBetweenWin', numSamplesHopBetweenWin, ...
    'fftLenCoeff', fftLenCoeff, ...
    'fftLen', fftLen, ...
    ...
    'SpectrumType', SpectrumType, ...
    'numBands', numBands, ...
    'WindowNormalization', WindowNormalization, ...
    'FilterBankNormalization', FilterBankNormalization, ...
    ...
    'numSpectrumsPerSpectrogram', numSpectrumsPerSpectrogram, ...
    'overlapCoeffSpectrograms', overlapCoeffSpectrograms, ...
    'numSpectrumsOverlapBetweenSpectrograms', numSpectrumsOverlapBetweenSpectrograms, ...
    'numSpectrumsHopBetweenSpectrograms', numSpectrumsHopBetweenSpectrograms, ...
    'segmentLen', segmentLen, ...
    'numSamplesHopBetweenSpectrograms', numSamplesHopBetweenSpectrograms ...
    );

end
