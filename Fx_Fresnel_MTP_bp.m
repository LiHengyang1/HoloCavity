function uin_ = Fx_Fresnel_MTP_bp(uout_, paras)
uout_ = uout_ .* conj(paras.FresnelCompensate);
uout_ = uout_ * paras.ratio / length(uout_);
uin_ = (paras.omegaX' * uout_ * paras.omegaX') .* conj(paras.Fresnelcore);
end