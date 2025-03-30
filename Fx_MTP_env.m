function [Parameters] = Fx_MTP_env(N, lambda, z0, L0, Limg, GPU)

ratio = Limg / (N * lambda * z0 / L0);
k = 2 * pi / lambda;
gridbase = ([0 : N - 1] - (N - 1) / 2).';
[U,V] = meshgrid(gridbase,gridbase);
pixel_L0 = L0 / N;
xx0 = U .* pixel_L0;
yy0 = V .* pixel_L0;
Parameters.Fresnelcore = exp(1i * k / 2 / z0 * (xx0.^2 + yy0.^2));

pixel_img = lambda * z0 / L0 * ratio;
xx1 = (U) .* pixel_img;
yy1 = (V) .* pixel_img;
Parameters.FresnelCompensate = exp(1i * k / 2 / z0 * (xx1.^2 + yy1.^2));

Parameters.ratio = ratio;

m = linspace(0,N-1,N) / N - 0.5 + 0.5 /N;
meshM = meshgrid(m,m);
Parameters.omegaX = exp(-1i * 2 * pi * meshM .* meshM.' * N * ratio);

if GPU
    Parameters.Fresnelcore = gpuArray(double(Parameters.Fresnelcore));
    Parameters.FresnelCompensate = gpuArray(double(Parameters.FresnelCompensate));
    Parameters.ratio = gpuArray(double(Parameters.ratio));
    Parameters.omegaX = gpuArray(double(Parameters.omegaX));
end
end