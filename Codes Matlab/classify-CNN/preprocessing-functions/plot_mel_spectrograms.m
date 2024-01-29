%%

close all;

len = length(data);
time = (0 : len - 1) / fs;

spectrum_type = ["power" "magnitude"];
win_norm = [true false];
filter_bank_norm = ["bandwidth" "area" "none"];

figure;
plot(time, data);
axis tight

for a = 1 : 2
    for b = 1 : 2
        for c = 1 : 3
            
            figure;
%             subplot(2,1,1);
%             plot(time, data);
%             axis tight
%             
%             subplot(2,1,2);
            
            melSpectrogram(data,fs, ...
                'Window',hann(256, 'periodic'), ...
                'OverlapLength',128, ...
                'FFTLength',512, ...
                'NumBands',64, ...
                'SpectrumType', spectrum_type(a), ...
                'WindowNormalization', win_norm(b), ...
                'FilterBankNormalization', filter_bank_norm(c) ...
                );
            colorbar off;
            title([spectrum_type(a), string(win_norm(b)), filter_bank_norm(c)]);
        end
    end
end