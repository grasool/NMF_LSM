function f = Dirac_delta(x, epsilon)
f=0.5*cos(atan(x./epsilon))*epsilon./(epsilon^2.+ x.^2);