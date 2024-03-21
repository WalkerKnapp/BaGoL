function Xi = sampleGam(NPoints,K,Xi,Alpha,Beta)
%sampleGam() infers parameters of gamma prior on number of locs per emitters
%
%Here, the prior parameters are learned using a hierarchical Bayes skim.
%
% INPUT:
%   NPoints: Number of localizations within ROIs
%   K:       Number of found emitters within ROIs
%   Xi:      Current parameter values of the Prior
%   Alpha:   Shape parameter of Lambda hyper-prior (Default = 1)
%   Bets:    Scale parameter of Lambda hyper-prior (Default = 50)
%
% OUTPUT: 
%   Xi:  Updated shape and scale parameters for gamma prior on number
%            of emitters

% Created by:
%   Mohamadreza Fazel (Lidke lab, 2022)
% 

Eta = Xi(1);
Gamma = Xi(2);

if nargin < 4
    Alpha = 1;
end
if nargin < 5
    Beta = 50;
end
Alpha_Prop = 3000;
Alpha_Prop_RandGs = randg(Alpha_Prop, 10*2) / Alpha_Prop;

Ind = NPoints > 0;
NPoints = NPoints(Ind);
K = K(Ind);

%10 samples are taken in a row and only the last one is returned
for ii = 1:10
    
    %Sample Eta
    Eta_Prop = Alpha_Prop_RandGs(2*ii) * Eta;

    LogLikeR = sum(logGammaRatioK(NPoints, K*Eta_Prop, K*Eta, Gamma));
    
    LogPriorR = logGammaRatioX(Eta_Prop, Eta, Alpha, Beta);
    LogPropR = logGammaRatioXTheta(Eta, Eta_Prop, Alpha_Prop, Eta_Prop/Alpha_Prop, Eta/Alpha_Prop);

    if LogLikeR + LogPriorR + LogPropR > log(rand())
       Eta = Eta_Prop; 
    end

    %Sample Gamma
    Gamma_Prop = Alpha_Prop_RandGs(2*ii + 1) * Gamma;
    
    LogLikeR = sum(logGammaRatioTheta(NPoints, K*Eta, Gamma_Prop, Gamma));

    LogPriorR = logGammaRatioX(Gamma_Prop, Gamma, Alpha, Beta);
    LogPropR = logGammaRatioXTheta(Gamma, Gamma_Prop, Alpha_Prop, Gamma_Prop/Alpha_Prop, Gamma/Alpha_Prop);

    if LogLikeR + LogPriorR + LogPropR > log(rand())
       Gamma = Gamma_Prop; 
    end

end
Xi(1) = Eta;
Xi(2) = Gamma;

end

function R = logGammaRatioK(x, k1, k2, theta)
    % An analytical solution to the log of the ratio of two Gamma PDFs
    % with the same `x` and `theta` parameters, and differing `k`.
    R = (k1 - k2) .* log(x) + (k2 - k1) .* log(theta) + gammaln(k2) - gammaln(k1);
end

function R = logGammaRatioTheta(x, k, theta1, theta2)
    % An analytical solution to the log of the ratio of two Gamma PDFs
    % with the same `x` and `k` parameters, and differing `theta`.
    R = k * log(theta2 ./ theta1) + x./theta2 - x./theta1;
end

function R = logGammaRatioX(x1, x2, k, theta)
    % An analytical solution to the log of the ratio of two Gamma PDFs
    % with the same `k` and `theta` parameters, and differing `x`.
    R = (k - 1) .* log(x1 ./ x2) + (x2 ./ theta) - (x1 ./ theta);
end

function R = logGammaRatioXTheta(x1, x2, k, theta1, theta2)
    % An analytical solution to the log of the ratio of two Gamma PDFs
    % with the same `k` parameter and different `x` and `theta`.
    R = k .* log(theta2 ./ theta1) + (k - 1) * log(x1 ./ x2) + (x2 ./ theta2) - (x1 ./ theta1);
end