function coughEventLocations = coughIndx2eventLocation(ADS, lbls_pred_dec, adsSpecs, win_info)

% pre-allocation
indx = 0;
[num_datasets, ~] = size(ADS);
coughEventLocations = [];
plot_indx = 0;

% loop over all audio file in audio set

for i = 1 : num_datasets
    
    ads = ADS{i, 1}; % audioDataStore in cell
    
    reset(ads);
    while hasdata(ads)
        
        [audioIn, ~] = read(ads);
        
        indx = indx + 1;
        sig_tail_indx = adsSpecs.sig_end_indx(indx);
        x = double(adsSpecs.start_indx(indx) : adsSpecs.end_indx(indx))';
        lbls_pred_dec_per_subj = double(lbls_pred_dec(x));
        
        % decimated sampling frequency
        fs_dec = win_info.fs * length(lbls_pred_dec_per_subj) / ...
            length(audioIn(1 : sig_tail_indx - win_info.segment_len));
        
        % find start and end points of cough event
        diff_event = diff([0 ; lbls_pred_dec_per_subj ; 0]);
        up_indx_event_dec = find(diff_event > 0);
        down_indx_event_dec = find(diff_event < 0) - 1;
        
        % convert up/down indices to original location based on fs
        lbls_pred = zeros(size(audioIn));
        up_indx_event = round(up_indx_event_dec / fs_dec * win_info.fs);
        down_indx_event = round(down_indx_event_dec / fs_dec * win_info.fs);
        for j = 1 : length(up_indx_event)
            lbls_pred(up_indx_event(j) : down_indx_event(j)) = 1;
        end
        
        % concatenate all auto-seg indices into one cell
        coughEventLocations = cat(1, coughEventLocations, ...
            {up_indx_event, down_indx_event, lbls_pred});
        
%         % plot some results for sanity check
% %         plot_indx = plot_indx + 1;
%         if plot_indx <= 40
%             plot_sig(audioIn, lbls_pred, win_info);
%             plot_indx = plot_indx + 1;
%         end
    end
end

end

