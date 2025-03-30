function Uout = Fx_Fresnel_MTP(Uin, paras)
Uout = paras.omegaX * (Uin .* paras.Fresnelcore) * paras.omegaX;
Uout = Uout * paras.ratio / length(Uout);
Uout = Uout .* paras.FresnelCompensate;
end