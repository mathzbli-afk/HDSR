% ——Original SR_Fun2.m; renamed function——
function Fval = sr1d_fun(t,x,s,a,b)
% 1-D bistable SR equation right-hand side
Fval = a*x - b*x.^3 + s;
end