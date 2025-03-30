function [Uf] = Fx_fresnelDFFT(U0,N,z0,lambda,L0)
%  DFFT方法的角谱衍射，能量保持真实情况（已经补偿，但是物理上散出去的光能量就真的散出去了），如果不单独传递参数则认为使用物象光场大小、采样数完全相同
%  input: U0光场  N一行（一列）采样数，要求必须是正方形   z0衍射距离   lambda波长   L0物光场尺度
%  output：Uf真实复振幅光场   If真实能量场强
    k = 2 * pi / lambda;  

    Uf = fftshift(fft2(U0));
    gridbase = ([0 : N - 1] - (N - 1) / 2).';
    gridbase = (1 / L0) * gridbase;	 	
    [U,V] = meshgrid(gridbase,gridbase);
    angularspectrumcore = exp(1i * k * z0 * (1 - lambda.^2 * 0.5 * (U.^2 + V.^2)));
    Uf = Uf .* angularspectrumcore;  
    Uf = ifft2(ifftshift(Uf));
end