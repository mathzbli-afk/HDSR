function eps_col = fox_exponential_colored_noise(K, dt,sigma,tau_c)
% generate colored noise with  Fox et al. (1988)  Box-Muller method

    % Fox parameter lambda = 1 / correlation time
    lambda = 1 / tau_c;
    D = sigma^2 / lambda;
    % E = exp(-lambda * Delta t)
    E = exp(-lambda * dt);

    % initial
    eps_col = zeros(K, 1);

    % ------------------------
    % Initial value using Box-Muller
    % epsilon = sqrt(-2*D*lambda*ln(m)) * cos(2*pi*n);m、n are random
    % ------------------------
    m0 = max(rand, realmin);  % avoid log(0)
    n0 = rand;
    eps_col(1) = sqrt(-2 * D * lambda * log(m0)) * cos(2*pi*n0);

    % ------------------------
    % Generate subsequent noise
    % ------------------------
    for k = 1:K-1
        a = max(rand, realmin);
        b = rand;
        h = sqrt(-2 * D * lambda * (1 - E^2) * log(a)) * cos(2*pi*b);
        eps_col(k+1) = E * eps_col(k) + h;
    end
end
 