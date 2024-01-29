function [F1_score_per_event, sensitivity_per_event, PPV_per_event] = ...
    seg_comp_per_event(lblsPred, lblsTrue, overlapRatioThreshold)

% convert labels to double
lblsPred = double(lblsPred) - 1;
lblsTrue = double(lblsTrue) - 1;

% find start and end points of cough event for true labels
diff_true = diff([0 ; lblsTrue ; 0]);
up_indx_true = find(diff_true > 0);
down_indx_true = find(diff_true < 0);
down_indx_true(end) = down_indx_true(end) - ...
    (down_indx_true(end) > length(lblsTrue));

% check all possible cases for predictions
% 1. all are nonCough
% 2. all are Cough
% 3. something in between

switch sum(lblsPred == 1) % how many segments are Cough
    
    case 0 % non is Cough
        
        TP_event = 0;
        FP_event = 0;
        FN_event = 1;
        
        % calculate scores per event
        sensitivity_per_event =   TP_event / (TP_event + FN_event) * 100;
        PPV_per_event =           TP_event / (TP_event + FP_event) * 100;
        F1_score_per_event =      TP_event ./ (TP_event + (FP_event + FN_event)/2) * 100;
        
        return;
        
    otherwise % all are Cough or something in between
        
        % OK. do nothing
            
end

% find start and end points of cough event for predicted labels
diff_pred = diff([0 ; lblsPred ; 0]);
up_indx_pred = find(diff_pred > 0);
down_indx_pred = find(diff_pred < 0);
down_indx_pred(end) = down_indx_pred(end) - ...
    (down_indx_pred(end) > length(lblsPred));
down_indx_pred = down_indx_pred - 1;

% pre-allocation
TP_event = 0;
FP_event = 0;
FN_event = 0;

% num_false_FN_12 = 0; % for reduction of extra FN for phase 1&2 detection


% loop over predicted labels

for i = 1 : length(up_indx_pred)
    
    pred_duration = sum(lblsPred(up_indx_pred(i) : down_indx_pred(i)));
    true_duration = sum(lblsTrue(up_indx_pred(i) : down_indx_pred(i)));
    overlap_ratio = true_duration / pred_duration;
    
    if overlap_ratio >= overlapRatioThreshold
        
        TP_event = TP_event + 1; % TP + 1
        
    else
        FP_event = FP_event + 1; % FP + 1
    end
end


% loop over true labels

for i = 1:length(up_indx_true)
    
    pred_duration = sum(lblsPred(up_indx_true(i) : down_indx_true(i)));
    true_duration = sum(lblsTrue(up_indx_true(i) : down_indx_true(i)));
    overlap_ratio = pred_duration / true_duration;
    
    if overlap_ratio < overlapRatioThreshold
        
        FN_event = FN_event + 1; % FN + 1
        
        if pred_duration > 0
            TP_event = TP_event - 1; % TP - 1
        end
    end
end

% calculate scores per event
sensitivity_per_event   = TP_event / (TP_event + FN_event) * 100;
PPV_per_event           = TP_event / (TP_event + FP_event) * 100;
F1_score_per_event      = TP_event ./ (TP_event + (FP_event + FN_event)/2) * 100;

end

