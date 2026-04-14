function Fval = sr4d_fun(t,Y,s,a,b,c,d,e,f,g,h)
% Right-hand side of 4-D stochastic resonance system
% Mathematical form:
%   y' = a*(x - y) + b*w
%   x' = c*y + x + d*y*z + s
%   z' = e*y*x + f*z
%   w' = g*y + h*x
y = Y(1);
x = Y(2);
z = Y(3);
w = Y(4);
Fval(1) = a*(x - y) + b*w;
Fval(2) = c*y + x + d*y*z + s;
Fval(3) = e*y*x + f*z;
Fval(4) = g*y + h*x;
Fval = Fval(:);
end
