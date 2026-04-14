
% sr1d_solver.m
% ——Original TRunge_Kutta222.m; renamed file & function name consistent——
function [t,x] = sr1d_solver(t,x0,Fs,S,a,b,K)
% ---------- RK / Euler solver for 1-D stochastic resonance ----------
h = 1/Fs;
t(1) = 0;          % Starting point
x(:,1) = x0;       % Can be vector

for i = 1:K-1
    t(i+1) = t(i) + h;
    s = S(i);

    % ------- Euler Method -------
    %x(:,i+1) = x(:,i) + h*sr1d_fun(t(i),x(:,i),s,a,b);

    % ------- Runge-Kutta Method -------

    k1 = sr1d_fun(t(i),          x(:,i),            s,a,b);
    k2 = sr1d_fun(t(i)+h/2,      x(:,i)+h*k1/2,     s,a,b);
    k3 = sr1d_fun(t(i)+h/2,      x(:,i)+h*k2/2,     s,a,b);
    k4 = sr1d_fun(t(i)+h,        x(:,i)+h*k3,       s,a,b);
    x(:,i+1) = x(:,i)+h*(k1+2*k2+2*k3+k4)/6;

end
end
