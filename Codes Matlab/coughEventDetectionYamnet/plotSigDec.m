function plotSigDec(audioSignal, time, ads, fsDec, lblsPredDecPerSubj, addTrueLbls, lblsTrueDecPerSubj)

% make sure that audioSignal and time have same length
audioSignal = audioSignal(1 : min(length(audioSignal), length(time)));
time = time(1 : min(length(audioSignal), length(time)));

timeDec = (0 : length(lblsPredDecPerSubj) - 1)  / fsDec;

% plot info
fontSize = 10;

% Create figure
figure;

switch addTrueLbls
    
    case 'yes'
        
        plot(time, audioSignal, 'linewidth', 0.2, ...
            'Color',[0.502 0.502 0.502]);
        hold on;
        plot(timeDec, lblsPredDecPerSubj, 'LineWidth', 5, 'Color', [0 0 1]);
        plot(timeDec, lblsTrueDecPerSubj, 'LineWidth', 2.5, 'Color', [0 1 1]);
        hold off;
        
        title(ads.dataset + ' ' + ads.subjectId, 'Signal with labels');
        xlabel('Time [sec]');
        axis([ ...
            timeDec(min(find(lblsTrueDecPerSubj == 1, 1), find(lblsTrueDecPerSubj == 1, 1))) - 0.5, ...
            timeDec(max(find(lblsTrueDecPerSubj == 1, 1, "last"), find(lblsTrueDecPerSubj == 1, 1, "last"))) + 0.5, -1, 1]);
        legend('signal', 'automatic labels', 'manual labels', ...
            'location', 'southeast', ...
            'FontSize', fontSize);
        
    case 'no'
        
        plot(time, audioSignal, 'linewidth', 0.2, ...
            'Color',[0.502 0.502 0.502]);
        
        hold on;
        plot(timeDec, lblsPredDecPerSubj, 'LineWidth', 5, 'Color', [0 0 1]);
        hold off;
        
        axis tight;
        
        title(ads.dataset + ' ' + ads.subjectId);
        xlabel('Time [sec]');
        legend('signal', 'automatic labels', ...
            'location', 'bestoutside', ...
            'FontSize', fontSize);
end

    ax = gca;
    ax.FontSize = 15;
    ax.YTick = [-1 -0.5 0 0.5 1];
    
end

