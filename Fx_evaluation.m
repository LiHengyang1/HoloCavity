function [Fidelity] = Fx_evaluation(field, source)


Fidelity = sum(conj(field) .* source, "all") ./ (sum(abs(field).^2,"all") * sum(abs(source).^2,"all")).^0.5;
Fidelity = abs(Fidelity).^2;
end