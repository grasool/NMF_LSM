function f=normalize01(f)
% Normalize to the range of [0,1]

fmin  = min(f(:));
fmax  = max(f(:));
de=fmax-fmin;
f = (f-fmin)/(de+(de==0)*eps);  % Normalize f to the range [0,1]

