function Fval = sr4d_fun(t,Y,s,a,b,c,d,e,f,g,h)
% State order used in implementation:
% Y = [y; x; z; w].
% The output signal x(t) corresponds to Y(2).
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
