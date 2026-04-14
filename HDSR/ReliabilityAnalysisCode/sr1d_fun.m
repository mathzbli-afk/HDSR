function Fval = sr1d_fun(t,x,s,a,b)
% Right-hand side of 1-D bistable stochastic resonance equation
% Mathematical form: dx/dt = a*x - b*x^3 + s
Fval = a*x - b*x.^3 + s;
end
