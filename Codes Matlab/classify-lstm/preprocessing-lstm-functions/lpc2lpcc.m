function lpcc = lpc2lpcc(lpc_coeffs, predErrorVar)

[num_rows, num_cols] = size(lpc_coeffs);
num_coeffs = num_rows - 1;

lpcc = zeros(num_coeffs, num_cols);
lpcc(1, :) = log(predErrorVar);
% lpcc(1, :) = log(num_coeffs);

for m = 1 : num_coeffs
    
    for k = 1 : m - 1
        lpcc(m , :) = lpcc(m , :) - (m - k) .* lpc_coeffs(k , :) .* lpcc(m - k , :);
    end
    
    lpcc(m , :) = (1/m) .* lpcc(m , :) - lpc_coeffs(m , :);
end

end

