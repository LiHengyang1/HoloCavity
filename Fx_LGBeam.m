function beamProfile = Fx_LGBeam(l,p,omega,dx,pm)
xxx = linspace(-0.5*dx*pm + 0.5*dx, 0.5*dx*pm - 0.5*dx, pm);
[xx,yy] = meshgrid(xxx,xxx);
[theta, r] = cart2pol(xx,yy);
normed_r = r ./ omega;

syms x
y = laguerreL(p,l,x);
LagerTerm = double(subs(y,x,2 * normed_r.^2));
beam0 = sqrt(2 * factorial(p) / pi / (abs(l) + p)) * (-1).^p / ...
    omega * (sqrt(2) * normed_r.^2 ./ r).^abs(l);
beam1 = LagerTerm .* exp(- normed_r.^2) .* exp(-1i * l * theta);
beam1 = beam1 .* beam0;
beamProfile = beam1 ./ max(abs(beam1),[],"all");

end

