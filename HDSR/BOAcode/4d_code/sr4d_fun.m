% 4D SR dynamics, with a small bug: x and y order not consistent with the paper
function Fval = sr4d_fun(t,Y,s,a,b,c,d,e,f,g,h)
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