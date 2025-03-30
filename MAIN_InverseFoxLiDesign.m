% This is the main code for "HoloCavity: Pure-phase Holographic On-demand Laser through Inverse Fox-Li Design".
% The design example here is: LG2,2 as desired mode, LG-2,2, LG1,1, and LG2,0 as undesired modes.
% A Nvidia graphic card with more than 6GB graphic RAM is strongly prefered. ELse, please comment out all codes about "gpuArray".
% Any questions please contact Hengyang Li via d202180830@hust.edu.cn
% Coded and tested on Matlab R2024b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear;
close all;
gpuDevice(1);
reset(gpuDevice);
%% Basic physical quantity definition
lambda = 1064e-6;
k = 2 * pi / lambda;
dx = 12.5e-3;
pm = 500;
pn = 500;
lx = pm * dx;
ly = pn * dx;
z1 = 456;
z2 = 274;
z3 = 590;
x = linspace(-lx / 2 + dx / 2,lx / 2 - dx / 2, pm);
[x,y] = meshgrid(x,x);
[theta,r] = cart2pol(x,y);
theta = theta + pi;
%%  Read desired mode and undesired modes
load LG22.mat
Q_RA = A1;
load LG22.mat
Q_CO1 = conj(A1);
load LG11.mat
Q_CO2 = (A1);
load LG20.mat
Q_CO3 = (A1);
IntenLoss = 0.99;    % IntenLoss is the alpha in article (intensity loss allowed in a round trip) 
%% Initial guess of the holograms
PHI1 = k * r.^2 / 2 / 9000;
PHI2 = k * r.^2 / 2 / 9000;
%% Preparing MTP propagation environment & transfer matrices to gRAM
pa1 = Fx_MTP_env(pm, lambda, z1, lx, lx, 1);
pa2 = Fx_MTP_env(pm, lambda, z2, lx, lx, 1);
pa3 = Fx_MTP_env(pm, lambda, z3, lx, lx, 1);
veloc1 = zeros(pm,pn);
veloc2 = zeros(pm,pn);

veloc1 = gpuArray(single(veloc1));
veloc2 = gpuArray(single(veloc2));
Q_RA = gpuArray(single(Q_RA));
Q_CO1 = gpuArray(single(Q_CO1));
Q_CO2 = gpuArray(single(Q_CO2));
Q_CO3 = gpuArray(single(Q_CO3));
PHI1 = gpuArray(single(PHI1));
PHI2 = gpuArray(single(PHI2));
%% Inverse Fox-Li design (optimization section)
LR1 = 50e-2;
LR2 = 50e-2;
cutoffRatio = 0;    %suppression weight
momentum = 0.93;    % momentum in SGDM
TVratio = 0.0001;    % weight of TV regularization
tic
for ii = 1:2000
    %%%%%%%%%%%%%%%%%%%%%%%%%% desired mode LG2,2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_SLM1_pre = Fx_Fresnel_MTP(Q_RA, pa1);
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    DIFF_RA = abs(U_OC_dejavu - IntenLoss * Q_RA).^2;
    Loss_RA(ii) = sum(DIFF_RA,"all");

    U_OC_dejavu_ = 2 * (U_OC_dejavu - IntenLoss * Q_RA);
    U_SLM2_post_ = Fx_Fresnel_MTP_bp(U_OC_dejavu_, pa3);
    U_SLM2_pre_ = U_SLM2_post_ .* exp(-1i * PHI2);
    PHI2_RA_ = imag(U_SLM2_post_ .* conj(U_SLM2_post));
    U_SLM1_post_ = Fx_Fresnel_MTP_bp(U_SLM2_pre_, pa2);
    PHI1_RA_ = imag(U_SLM1_post_ .* conj(U_SLM1_post));

    FideRA(ii) = Fx_evaluation(U_OC_dejavu, Q_RA);    % OK to comment out
    %%%%%%%%%%%%%%%%%%%%%%%%%% TV regularization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TV1x = PHI1 - circshift(PHI1,1,1);
    TV1y = PHI1 - circshift(PHI1,1,2);
    TV1xs = abs(TV1x).^2;
    TV1ys = abs(TV1y).^2;
    TV2x = PHI2 - circshift(PHI2,1,1);
    TV2y = PHI2 - circshift(PHI2,1,2);
    TV2xs = abs(TV2x).^2;
    TV2ys = abs(TV2y).^2;
    TVnorm(ii) = TVratio * sum((TV1xs + TV1ys) + (TV2xs + TV2ys),"all");

    TV1x_ = 2 * TVratio * TV1x;
    TV1y_ = 2 * TVratio * TV1y;
    TV2x_ = 2 * TVratio * TV2x;
    TV2y_ = 2 * TVratio * TV2y;
    TVPHI1_ = (TV1x_ + TV1y_ - circshift(TV1x_,-1,1) - circshift(TV1y_,-1,2));
    TVPHI2_ = (TV2x_ + TV2y_ - circshift(TV2x_,-1,1) - circshift(TV2y_,-1,2));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% undesired mode LG-2,2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    U_SLM1_pre = Fx_Fresnel_MTP(Q_CO1, pa1);
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    DIFF_CO = abs(U_OC_dejavu - Q_CO1).^2;
    Loss_CO1(ii) = sum(DIFF_CO,"all");

    Loss_CO0_ = -1 * cutoffRatio * Loss_CO1(ii).^(-2);
    U_OC_dejavu_ = 2 * (U_OC_dejavu - Q_CO1) * Loss_CO0_;
    U_SLM2_post_ = Fx_Fresnel_MTP_bp(U_OC_dejavu_, pa3);
    U_SLM2_pre_ = U_SLM2_post_ .* exp(-1i * PHI2);
    PHI2_CO_1 = imag(U_SLM2_post_ .* conj(U_SLM2_post));
    U_SLM1_post_ = Fx_Fresnel_MTP_bp(U_SLM2_pre_, pa2);
    PHI1_CO_1 = imag(U_SLM1_post_ .* conj(U_SLM1_post));

    FideCO1(ii) = Fx_evaluation(U_OC_dejavu, Q_CO1);    % OK to comment out
    %%%%%%%%%%%%%%%%%%%%%%%%%% undesired mode LG1,1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_SLM1_pre = Fx_Fresnel_MTP(Q_CO2, pa1);
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    DIFF_CO = abs(U_OC_dejavu - Q_CO2).^2;
    Loss_CO2(ii) = sum(DIFF_CO,"all");

    Loss_CO0_ = -1 * cutoffRatio * Loss_CO2(ii).^(-2);
    U_OC_dejavu_ = 2 * (U_OC_dejavu - Q_CO2) * Loss_CO0_;
    U_SLM2_post_ = Fx_Fresnel_MTP_bp(U_OC_dejavu_, pa3);
    U_SLM2_pre_ = U_SLM2_post_ .* exp(-1i * PHI2);
    PHI2_CO_2 = imag(U_SLM2_post_ .* conj(U_SLM2_post));
    U_SLM1_post_ = Fx_Fresnel_MTP_bp(U_SLM2_pre_, pa2);
    PHI1_CO_2 = imag(U_SLM1_post_ .* conj(U_SLM1_post));

    FideCO2(ii) = Fx_evaluation(U_OC_dejavu, Q_CO2);    % OK to comment out
    %%%%%%%%%%%%%%%%%%%%%%%%%% undesired mode LG2，0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    U_SLM1_pre = Fx_Fresnel_MTP(Q_CO3, pa1);
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    DIFF_CO = abs(U_OC_dejavu - Q_CO3).^2;
    Loss_CO3(ii) = sum(DIFF_CO,"all");

    Loss_CO0_ = -1 * cutoffRatio * Loss_CO3(ii).^(-2);
    U_OC_dejavu_ = 2 * (U_OC_dejavu - Q_CO3) * Loss_CO0_;
    U_SLM2_post_ = Fx_Fresnel_MTP_bp(U_OC_dejavu_, pa3);
    U_SLM2_pre_ = U_SLM2_post_ .* exp(-1i * PHI2);
    PHI2_CO_3 = imag(U_SLM2_post_ .* conj(U_SLM2_post));
    U_SLM1_post_ = Fx_Fresnel_MTP_bp(U_SLM2_pre_, pa2);
    PHI1_CO_3 = imag(U_SLM1_post_ .* conj(U_SLM1_post));

    FideCO3(ii) = Fx_evaluation(U_OC_dejavu, Q_CO3);    % OK to comment out
    %%%%%%%%%%%%%%%%%%%%%%%%%% Delayed suppressing  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ii > 400
        cutoffRatio =  4000 * (ii.^0.5) - 80000;    %gradually increasing suppression weight
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%% SGDM Optimizer  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PHI1_CO_ = PHI1_CO_1 + PHI1_CO_2 + PHI1_CO_3;
    PHI2_CO_ = PHI2_CO_1 + PHI2_CO_2 + PHI2_CO_3;
    fang1 = - PHI1_RA_ - PHI1_CO_ - TVPHI1_ + momentum * veloc1;
    fang2 = - PHI2_RA_ - PHI2_CO_ - TVPHI2_ + momentum * veloc2;
    PHI1 = PHI1 + LR1 * fang1;
    PHI2 = PHI2 + LR2 * fang2;
    if ii > 10
        veloc1 = fang1;
        veloc2 = fang2;
    end
    ii
end
toc
%% Cascaded holograms
PHI1 = gather(PHI1);
PHI2 = gather(PHI2);
PHI1(1,1) = pi;
PHI1(1,2) = -pi;
figure
imagesc(angle(exp(1i * PHI1)))
colormap(othercolor('BuOr_12'))
title('Hologram1')
PHI2(1,1) = pi;
PHI2(1,2) = -pi;
figure
imagesc(angle(exp(1i * PHI2)))
colormap(othercolor('BuOr_12'))
title('Hologram2')
%% Single round trip tomography
sliceZ = 20;
slicenum1 = z1 / sliceZ;
slicenum2 = z2 / sliceZ;
slicenum3 = z3 / sliceZ;
for ii = 1:slicenum1
    UUU1(:,:,ii) = Fx_fresnelDFFT(Q_RA,pm,(ii-1) * sliceZ,lambda,ly);
end
UUUUUU1 = Fx_fresnelDFFT(Q_RA,pm,z1,lambda,ly);
UUUUUU1 = UUUUUU1 .* exp(1i * PHI1);
for ii = 1:slicenum2
    UUU2(:,:,ii) = Fx_fresnelDFFT(UUUUUU1,pm,(ii-1) * sliceZ,lambda,ly);
end
UUUUUU2 = Fx_fresnelDFFT(UUUUUU1,pm,z2,lambda,ly);
UUUUUU2 = UUUUUU2 .* exp(1i * PHI2);
for ii = 1:slicenum3 + 1
    UUU3(:,:,ii) = Fx_fresnelDFFT(UUUUUU2,pm,(ii-1) * sliceZ,lambda,ly);
end
UUU = cat(3,UUU1,UUU2,UUU3);
UUU = gather(UUU);
figure;
sliceViewer(abs(UUU).^2)
colormap("turbo")
title('Single round trip tomography____Intensity')
figure;
sliceViewer(angle(UUU))
colormap(othercolor('BuOr_12'))
title('Single round trip tomography____Phase')
%% Chiral mode suprression tomography
OCfield_RA = zeros(pm,pn,200);
for ii = 1:200
    if ii == 1
        U_SLM1_pre = Fx_Fresnel_MTP(Q_CO1, pa1);
    else
        U_SLM1_pre = Fx_Fresnel_MTP(U_OC_dejavu, pa1);
    end
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    OCfield_RA(:,:,ii) = gather(U_OC_dejavu);
    ii
end
for ii = 1:200
    OCfield_RA(:,:,ii) = OCfield_RA(:,:,ii) ./ abs(max(max(OCfield_RA(:,:,ii))));
end
figure;
sliceViewer(abs(OCfield_RA(:,:,1:end)).^2);
colormap("turbo")
title('Chiral mode suprression tomography____Intensity')
figure;
sliceViewer(angle(OCfield_RA(:,:,1:end)));
colormap(othercolor('BuOr_12'))
title('Chiral mode suprression tomography____Phase')
%% LG55 suprression tomography
load LG55.mat
Q_CO = A1;
for ii = 1:200
    if ii == 1
        U_SLM1_pre = Fx_Fresnel_MTP(Q_CO, pa1);
    else
        U_SLM1_pre = Fx_Fresnel_MTP(U_OC_dejavu, pa1);
    end
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    OCfield_CO(:,:,ii) = U_OC_dejavu;
    ii
end
OCfield_CO = gather(OCfield_CO);
for ii = 1:200
    OCfield_CO(:,:,ii) = OCfield_CO(:,:,ii) ./ abs(max(max(OCfield_CO(:,:,ii))));
end
figure;
sliceViewer(abs(OCfield_CO).^2);
colormap("turbo")
title('LG55 suprression tomography____Intensity')
figure;
sliceViewer(angle(OCfield_CO));
colormap(othercolor('BuOr_12'))
title('LG55 suprression tomography____Phase')
%% Mode compitition tomography
Q_CO(:,:,1) = Fx_gaussianbeam(pm,pn,50,dx);
OCfield_CO = zeros(pm,pn,200);
for ii = 1:200
    if ii == 1
        U_SLM1_pre = Fx_Fresnel_MTP(Q_CO(:,:,1), pa1);
    else
        U_SLM1_pre = Fx_Fresnel_MTP(U_OC_dejavu, pa1);
    end
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    OCfield_CO(:,:,ii) = gather(U_OC_dejavu);
    ii
end
for ii = 1:200
    OCfield_CO(:,:,ii) = OCfield_CO(:,:,ii) ./ abs(max(max(OCfield_CO(:,:,ii))));
end
figure;
sliceViewer(abs(OCfield_CO(:,:,2:end)).^2);
colormap("turbo")
title('Mode compitition tomography____Intensity')
figure;
sliceViewer(angle(OCfield_CO(:,:,2:end)));
colormap(othercolor('BuOr_12'))
title('Mode compitition tomography____Phase')
%% position-shift error
PHI1q = circshift(PHI1,0,1);
PHI2q = circshift(PHI2,-4,1);
pa1 = Fx_MTP_env(pm, lambda, z1 + 0, lx, lx, 1);
pa2 = Fx_MTP_env(pm, lambda, z2 + 0, lx, lx, 1);
pa3 = Fx_MTP_env(pm, lambda, z3 + 0, lx, lx, 1);
for ii = 1:500
    if ii == 1
        U_SLM1_pre = Fx_Fresnel_MTP(Q_RA, pa1);
    else
        U_SLM1_pre = Fx_Fresnel_MTP(U_OC_dejavu, pa1);
    end
    U_SLM1_post = U_SLM1_pre .* exp(1i * PHI1q);
    U_SLM2_pre = Fx_Fresnel_MTP(U_SLM1_post, pa2);
    U_SLM2_post = U_SLM2_pre .* exp(1i * PHI2q);
    U_OC_dejavu = Fx_Fresnel_MTP(U_SLM2_post, pa3);
    OCfield_RA(:,:,ii) = U_OC_dejavu;
    ii
end
OCfield_RA = gather(OCfield_RA);
for ii = 1:500
    OCfield_RA(:,:,ii) = OCfield_RA(:,:,ii) ./ abs(max(max(OCfield_RA(:,:,ii))));
end
figure;
sliceViewer(abs(OCfield_RA).^2);
colormap("turbo")
title('position-shift error____Intensity')
figure;
sliceViewer(angle(OCfield_RA));
colormap(othercolor('BuOr_12'))
title('position-shift error____Phase')