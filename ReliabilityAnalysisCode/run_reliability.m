%% ===== Function =====
function [snri_curve,bias_curve, norm_bias_curve, success_total] = run_reliability(params, use4d, fj_vals, tau1)
    N = length(fj_vals);
    snri_curve = zeros(N,1);
    bias_curve      = zeros(N,1);
    norm_bias_curve = zeros(N,1);
    success_flags   = zeros(N,1);

    for j = 1:N
        fj = fj_vals(j);

        if use4d
            [~, SNRI_dB, fhat] = sr4d_eval_random(params, fj);
        else
            [~, SNRI_dB, fhat] = sr1d_eval_random(params, fj);
        end

        % SNRI-fj
        snri_curve(j) = SNRI_dB;

        % Frequency bias: delta fj = fj - f^j 
        bias_curve(j) = fj - fhat;

        % Normalized frequency bias: |f_j - fhat_j| / f_j
        norm_bias_curve(j) = abs(fj - fhat) / fj;

        % Success criterion based on Figure 1:
        % inside the bias tolerance band [-tau1, +tau1]
        success_flags(j) = double(abs(bias_curve(j)) <= tau1);
    end

    success_total = mean(success_flags);
end