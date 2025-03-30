function COEF = Fx_ComplexFidelity(sig,ref,dx,pm,ValidRound)


lx = pm * dx;
ly = pm * dx;
x = linspace(-lx / 2 + dx / 2,lx / 2 - dx / 2, pm);
[x,y] = meshgrid(x,x);
[theta,r] = cart2pol(x,y);

figure; 
mesh(angle(sig));
figure; 
mesh(angle(ref));

torq0 = ref ./ sig;
torq0(r>ValidRound) = 0;
Ftorq = fftshift(fft2(torq0));

Basetorq = angle(Ftorq(pm/2+1,pm/2+1))
sig = sig .* exp(1i * Basetorq);
COEF = abs(sum(conj(ref) .* sig,"all") ./ sqrt(sum(abs(conj(ref)).^2,"all") * sum(abs(sig).^2,"all"))).^2;




