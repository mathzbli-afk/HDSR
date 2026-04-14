% RK4 solver for 4D SR
function [t,Y] = sr4d_solver(t,Y0,Fs,S,a,b,c,d,e,f,g,h,K)
hsteps = 1/Fs;
% hsteps=1;
t(1) = 0;                      % Starting time point
Y(:,1) = Y0;                   % Initial value assignment, can be a vector, but dimension must match
for i = 1:K - 1                % Numerical solution using Runge-Kutta method    
    t(i + 1) = t(i) + hsteps;    
    s = S(i);
    % ------- Runge-Kutta Method -------
    tt = t(i);
    YY = Y(:,i);
    % Update state using Euler method
    %Y(:,i + 1) = YY + hsteps * sr4d_fun(tt, YY, s, a, b, c, d, e, f, g, h);
    K1 = sr4d_fun(tt,YY,s,a,b,c,d,e,f,g,h);
    K2 = sr4d_fun(tt + hsteps/2,YY + hsteps*K1/2,s,a,b,c,d,e,f,g,h);     
    K3 = sr4d_fun(tt + hsteps/2,YY + hsteps*K2/2,s,a,b,c,d,e,f,g,h);     
    K4 = sr4d_fun(tt + hsteps,YY + hsteps*K3,s,a,b,c,d,e,f,g,h);     
    Y(:,i+1) = YY + hsteps*(K1 + 2*K2 + 2*K3 + K4)/6;
    
end
if any(isnan(Y(:)))
    warning('Solver produced NaN, system unstable.');
end

end