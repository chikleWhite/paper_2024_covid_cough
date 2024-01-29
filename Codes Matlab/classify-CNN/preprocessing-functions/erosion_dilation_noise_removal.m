function lbls_pred_dec = erosion_dilation_noise_removal(lbls_pred_dec, win_info)

% erotion
% kernel: nx1 vector

% fs_dec = 25.1; % obtained from observation
fs_dec = 16000 / win_info.numSamplesHopBetweenSpectrograms;
time_threshold = 0.15; % chosen time interval 
mask_half_len = round(time_threshold * fs_dec / 2); % half length of kernel
mask_len = 2 * mask_half_len + 1; % length of kernel

% extend labels based on kernel size
lbls_pred_dec_extended = [zeros(mask_half_len, 1) ; lbls_pred_dec ; zeros(mask_half_len, 1)];
lbls_pred_dec_extended_len = length(lbls_pred_dec_extended);

% put erosion results in separate vector
erosion_indices = ones(size(lbls_pred_dec_extended));

% loop over all labels with erosion kernel
for indx = 1 + mask_half_len : lbls_pred_dec_extended_len - mask_half_len
    
    % range for erosion decision
    mask = lbls_pred_dec_extended(indx - mask_half_len : indx + mask_half_len);
    
    % if at list 1 non-cough (zero) in kernel -> set middle indx to zero
    if sum(mask) < mask_len
        erosion_indices(indx) = 0;
    end
end

% remove extended indices from erosion results
erosion_indices([1 : mask_half_len, end - mask_half_len + 1 : end]) = [];

% update new non-cough (zero) segments based on erosion
lbls_pred_dec = lbls_pred_dec .* erosion_indices;

% dilation

% extend new labels based on kernel size
lbls_pred_dec_extended = [zeros(mask_half_len, 1) ; lbls_pred_dec ; zeros(mask_half_len, 1)];

% put dialation results in separate vector
dilation_indices = zeros(size(lbls_pred_dec_extended));

% loop over all labels with dilation kernel
for indx = 1 + mask_half_len : lbls_pred_dec_extended_len - mask_half_len
    
    mask = lbls_pred_dec_extended(indx - mask_half_len : indx + mask_half_len);
    
    if sum(mask) > 0
        dilation_indices(indx) = 1;
    end
end

% remove extended indices from dilation results
dilation_indices([1 : mask_half_len, end - mask_half_len + 1 : end]) = [];

% update new non-cough (zero) segments based on dilation
lbls_pred_dec = max(lbls_pred_dec, dilation_indices);

end

