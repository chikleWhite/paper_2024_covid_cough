%%

% pre-processing - cough event indices
[Features, trueLabels, adsSpecs] = preprocess_classification_lstm(ads);

%%

num_coughs = adsSpecs.end_indx - adsSpecs.start_indx + 1;

num_voca_pos = 0;
num_voca_neg = 0;
num_coughvid_pos = 0;
num_coughvid_neg = 0;

for i = 1 : length(num_coughs)
    
    switch adsSpecs.database(i)
        
        case "voca"
            
            switch string(adsSpecs.label(i))
                
                case "positive"
                    
                    num_voca_pos = num_voca_pos + num_coughs(i);
                    
                case "negative"
                    
                    num_voca_neg = num_voca_neg + num_coughs(i);
            end
        case "coughvid"
            
            switch string(adsSpecs.label(i))
                
                case "positive"
                    
                    num_coughvid_pos = num_coughvid_pos + num_coughs(i);
                    
                case "negative"
                    
                    num_coughvid_neg = num_coughvid_neg + num_coughs(i);
            end
    end
end