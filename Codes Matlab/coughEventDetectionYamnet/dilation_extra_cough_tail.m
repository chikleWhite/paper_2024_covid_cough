function lbls_pred_dec = dilation_extra_cough_tail(lbls_pred_dec)

% dialation
% kernel: nx1 vector

mask_half_len = 1; % half length of kernel

% put dialation results in separate vector
dilation_indices = zeros(size(lbls_pred_dec));

% loop over all labels with dilation kernel
for indx = 1 + mask_half_len : length(lbls_pred_dec)
    
    mask = lbls_pred_dec(indx - mask_half_len : indx);
    
    if sum(mask) > 0
        dilation_indices(indx) = 1;
    end
end

% update new segments based on dilation
lbls_pred_dec = max(lbls_pred_dec, dilation_indices);

end

