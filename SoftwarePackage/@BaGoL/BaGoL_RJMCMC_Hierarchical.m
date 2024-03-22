function [K,Mu_X,Mu_Y,Alpha_X,Alpha_Y,Z]=BaGoL_RJMCMC_Hierarchical(SMD,Area,...
               SigAlpha,PMove,NSamples,Xi,Mu_X,Mu_Y,Alpha_X,Alpha_Y)
%BaGoL_RJMCMC_Hierarchical BaGoL's core RJMCMC algorithm that takes
%NSamples samples from the posterior using RJMCMC and return the last
%sample to be used in the hierarchical Bayes approach.
% [Chain]=BaGoL.BaGoL_RJMCMC(SMD,Xi,MaxAlpha,PMove,NChain,NBurnin,DEBUG)
%
% This is the core BaGoL algorithm. It uses Reversible Jump Markov Chain 
% Monte Carlo to add, remove and move emitters, and to explore the 
% allocation of localizations to emitters.  
%
% The number of localizations per emitter distribution is parameterized by  
% either a Poisson or Gamma distribution function. For the Poisson 
% distribution, the parameter [lambda] is the mean number of localizations 
% per emitter. For the Gamma distribution, [k, theta] are the shape and 
% scale parameters. k*theta is the mean localizations per emitter. 
%
% A linear drift of individual emitters can be included by allowing
% MaxAlpha to be non-zero. Drift velocities are given a uniform prior for
% each dimension from -MaxAlpha to MaxAlpha. 
%
% The output chain is a structure array of the post burn-in states for the
% input subregion. Each element of the array contains fields 
% associated to a single accepted proposal. A description of the fields is
% given below. 
%
% INPUTS:
%    SMD:      SMD structure with the following fields:
%       X:     X localization coordinates. (nm) (Nx1)
%       Y:     Y localization coordinates. (nm) (Nx1)
%       X_SE:  X localization precisions.  (nm) (Nx1)
%       Y_SE:  Y localization precisions.  (nm) (Nx1)
%       FrameNum:   localization frame numbers. (Nx1)
%    Xi:       Loc./emitter params [lambda] (Poisson) or [k theta] (Gamma) 
%    SigAlpha: Sigma of drift velocity. (nm) (Default = 0)
%    PMove:    Probabilities of proposing different moves in RJMCMC:
%              [1] Move Mu, Alpha
%              [2] Reallocation of Z
%              [3] Add
%              [4] Remove
%              (1x4) (Default = [0.25, 0.25, 0.25, 0.25])
%    NChain:   Length of the chain after the burn in. (Default = 2000)
%    NBurnin:  Length of the chain for burn in. (Default = 3000)
%    DEBUG:    0 or 1. Show an animation of the chain. (Default = 0)
%
% OUTPUT:
%    Chain:    Structure array of post burn-in states of the RJMCMC Chain. 
%       N: Number of emitters (Scalar)
%       X: X coordinate of emitters (Kx1)
%       Y: Y coordinate of emitters (Kx1)
%       AlphaX: Corresponding X drift velocities (Kx1)
%       AlphaY: Corresponding Y drift velocities (Kx1)
%       ID: Allocation parameter representing assigning a localization to 
%           an emitter. The order is the same as SMD.X (see above) (Nx1) 
%
% CITATION: "Sub-Nanometer Precision using Bayesian Grouping of Localizations"
%           Mohamadreza Fazel, Michael J. Wester, Sebastian Restrepo Cruz,
%           Sebastian Strauss, Florian Schueder, Thomas Schlichthaerle, 
%           Jennifer M. Gillette, Diane S. Lidke, Bernd Rieger,
%           Ralf Jungmann, Keith A. Lidke
%

% Created by: 
%    Mohamadreza Fazel and Keith A. Lidke (Lidkelab 2020)

%DEBUG=0;

X_min = min(SMD.X-3*SMD.X_SE);
Y_min = min(SMD.Y-3*SMD.Y_SE);

if nargin<4
    PMove = [.25 .25 .25 .25]; %PMove = [Theta Z Birth Death]
end
if nargin<5
    NSamples = 10;
end

N=length(SMD.X);
if nargin<7
%Intial K Guess
    K=ceil(N/prod(Xi));
else
    K=length(Mu_X); 
end
if nargin<7
%Initial Locations
    Mu_X =SMD.X(randi(N,[1 K]))';
    Mu_Y =SMD.Y(randi(N,[1 K]))';
end
if nargin<9
%Initial Alphas
    Alpha_X = zeros([1 K]);
    Alpha_Y = zeros([1 K]);
end

% Initial probabilities for each loc-emitter pair
pair_probs = computePairProbs(SMD, Mu_X, Mu_Y, Alpha_X, Alpha_Y);

% Intial Allocation
[Z, Mu_X, Mu_Y, Alpha_X, Alpha_Y, pair_probs] = ...
    Gibbs_Z(pair_probs, Mu_X, Mu_Y, Alpha_X, Alpha_Y);

if N < 100
   LengN = 100; 
else
   LengN = N; 
end
%Calculating the Prior
if length(Xi)>1
    Gamma_K=Xi(1);
    Gamma_Theta=Xi(2);
   
    % A highly optimized analytical solution to both:
    % PR_addition = gammapdf(N, (x+1)*k, theta)/gammapdf(N, x*k, theta)
    % PR_removal = gammapdf(N, (x-1)*k, theta)/gammapdf(N, x*k, theta)
    C = Gamma_K * log(N/Gamma_Theta);
    gam = gammaln((0:LengN+1)*Gamma_K);

    base = gam(2:end-1);
    PR_addition = exp(base - gam(3:end) + C);
    PR_removal = exp(base - gam(1:end-2) - C);
else
    Pk=poisspdf(N,(0:LengN+1)*Xi); 

    PR_addition = Pk(3:end) ./ Pk(2:end-1);
    PR_removal = Pk(1:end-2) ./ Pk(2:end-1);
end

% Convert PMove to CDF to simplify sampling in loop
PMove = cumsum(PMove);

% Run Chain
for nn=1:NSamples
    %Get move type:
    JumpType = length(PMove)+1 - sum(rand < PMove);
    K = length(Mu_X);

    switch JumpType
        case 1  %Move Mu, Alpha 
            Mu_XTest=Mu_X;
            Mu_YTest=Mu_Y;
            Alpha_XTest=Alpha_X;
            Alpha_YTest=Alpha_Y;
            
            %Get new Mu and Alpha using Gibbs
            if SigAlpha>0
                for ID=1:K  
                   [Mu_XTest(ID),Alpha_XTest(ID)]=Gibbs_MuAlpha(ID,Z,SMD.X,SMD.FrameNum,SMD.X_SE,SigAlpha);
                   [Mu_YTest(ID),Alpha_YTest(ID)]=Gibbs_MuAlpha(ID,Z,SMD.Y,SMD.FrameNum,SMD.Y_SE,SigAlpha);
                end
            else
                Mu_XTest = Gibbs_Mu(K, Z, SMD.X, SMD.X_SE);
                Mu_YTest = Gibbs_Mu(K, Z, SMD.Y, SMD.Y_SE);
            end
            
            Mu_X = Mu_XTest;
            Mu_Y = Mu_YTest;
            Alpha_X = Alpha_XTest;
            Alpha_Y = Alpha_YTest;

            pair_probs = computePairProbs(SMD, Mu_X, Mu_Y, Alpha_X, Alpha_Y);

        case 2  %Reallocation of Z
            
            [Z, Mu_X, Mu_Y, Alpha_X, Alpha_Y, pair_probs] = ...
                Gibbs_Z(pair_probs, Mu_X, Mu_Y, Alpha_X, Alpha_Y);
                                    
        case 3  %Add
                        
%           Sample the Emitter location from SR data
            em_sample = randi(length(SMD.X));
            Xdraw = randn() .* SMD.X_SE(em_sample) + SMD.X(em_sample);
            Ydraw = randn() .* SMD.Y_SE(em_sample) + SMD.Y(em_sample);
            Mu_XTest = cat(2, Mu_X, Xdraw);
            Mu_YTest = cat(2, Mu_Y, Ydraw);     

            if SigAlpha>0
                Alpha_XTest = cat(2,Alpha_X,SigAlpha*randn());
                Alpha_YTest = cat(2,Alpha_Y,SigAlpha*randn());
            else
                Alpha_XTest = cat(2,Alpha_X,0);
                Alpha_YTest = cat(2,Alpha_Y,0);
            end

            new_emitter_probs = computePairProbs(SMD, Mu_XTest(end), Mu_YTest(end), Alpha_XTest(end), Alpha_YTest(end));
            draw_pdf = sum(new_emitter_probs);

            pair_probs_test = cat(2, pair_probs, new_emitter_probs);
                        
            %Prior Raio
            PR = PR_addition(K);
            
            LAlloc_Current = p_Alloc(pair_probs);
            LAlloc_Test = p_Alloc(pair_probs_test);
            AllocR = exp(LAlloc_Test-LAlloc_Current);
            
            %Posterior Ratio
            A = PR*AllocR/(Area*draw_pdf);
            
            Accept = isinf(LAlloc_Current) & LAlloc_Current < 0;
            
            if rand<A || Accept
                %Gibbs allocation
                [Z, Mu_X, Mu_Y, Alpha_X, Alpha_Y, pair_probs] = ...
                    Gibbs_Z(pair_probs_test, Mu_XTest, Mu_YTest, Alpha_XTest, Alpha_YTest);
            end
            
        case 4  %Remove
            
            if K==1 %Then update chain and return
                continue;
            end
            
            %pick emitter to remove:
            ID =randi(K);
            
            Mu_XTest = Mu_X;
            Mu_YTest = Mu_Y;
            Alpha_XTest = Alpha_X;
            Alpha_YTest = Alpha_Y;
            
            %Remove from list
            Mu_XTest(ID) = [];
            Mu_YTest(ID) = [];
            Alpha_XTest(ID) = [];
            Alpha_YTest(ID) = [];

            pair_probs_test = pair_probs(:, [1:ID-1 ID+1:K]);
            
            %Prior Raio
            PR = PR_removal(K);
            
            %Probability Ratio of Proposed Allocation and Current Allocation 
            LAlloc_Current = p_Alloc(pair_probs);
            LAlloc_Test = p_Alloc(pair_probs_test);
            AllocR = exp(LAlloc_Test-LAlloc_Current);
            
            %Posterior Ratio
            A = PR*AllocR;
            
            if rand<A
                %Gibbs allocation
                [Z, Mu_X, Mu_Y, Alpha_X, Alpha_Y, pair_probs] = ...
                    Gibbs_Z(pair_probs_test, Mu_XTest, Mu_YTest, Alpha_XTest, Alpha_YTest);
            end
                        
    end    
    
    DEBUG = 0;
    if DEBUG==1 %for testing
        figure(1111)
        scatter(SMD.X,SMD.Y,[],Z)
        hold on
        plot(Mu_X,Mu_Y,'ro','linewidth',4)
        set(gca,'Ydir','reverse')
        legend(sprintf('Jump: %g',nn))
        xlabel('X(nm)')
        ylabel('Y(nm)')
        hold off
        pause(.001)
    elseif DEBUG == 2
        RadiusScale = 2;
        CircleRadius = sqrt((SMD.X_SE.^2 + SMD.Y_SE.^2) / 2) * RadiusScale;
        figure(1111)
        for oo = 1:max(Z)
            ID = Z==oo;
            Theta = linspace(0, 2*pi, 25)';
            Theta = repmat(Theta,[1,sum(ID)]);
            CircleX = repmat(CircleRadius(ID)',[25,1]).*cos(Theta) + repmat(SMD.X(ID)',[25,1]);
            CircleY = repmat(CircleRadius(ID)',[25,1]).*sin(Theta) + repmat(SMD.Y(ID)',[25,1]);
            A=plot(CircleX,CircleY);
            if oo == 1;hold on;end
            if ~isempty(A)
                set(A,'color',A(1).Color);
            end
        end
        plot(Mu_X,Mu_Y,'ro','linewidth',4)
        legend(sprintf('Jump: %g',nn))
        xlabel('X(nm)')
        ylabel('Y(nm)')
       
        hold off
        pause(.001)
    end
end

% K may have changed in the last iteration of the loop, update it
K = length(Mu_X);

end

function [ZTest, Mu_X, Mu_Y, Alpha_X, Alpha_Y, pair_probs]=Gibbs_Z(pair_probs, Mu_X, Mu_Y, Alpha_X, Alpha_Y)
    % Sample a new "allocation" for each localization, using the
    % probability that it belongs to an emitter (stored in pair_probs) as
    % the probability of an allocation being chosen.
    K = length(Mu_X);

    P=pair_probs+eps;  % Adjusted up to account for pair_probs rows with all 0
    CDF=cumsum(P,2)./sum(P,2);

    % Chooses a random index into each row with weight from CDF
    random_choice = rand(size(CDF, 1),1);
    threshold = random_choice<(CDF+eps);

    % Check for emitters with no localizations
    first_localized = any(threshold(:, 1));  % Any loc passing the threshold in the first column has been assigned to the first emitter
    remaining_localized = any(diff(threshold, 1, 2), 1);  % Use diff to identify which emitter other localizations belong to
    if ~(first_localized && all(remaining_localized))
        unlocalized_emitters = ~[first_localized remaining_localized];

        threshold(:, unlocalized_emitters) = [];
        Mu_X(unlocalized_emitters) = [];
        Mu_Y(unlocalized_emitters) = [];
        Alpha_X(unlocalized_emitters) = [];
        Alpha_Y(unlocalized_emitters) = [];
        pair_probs(:, unlocalized_emitters) = [];

        K = length(Mu_X);
    end

    ZTest=K+1-sum(threshold,2);
end

function [Mu,Alpha]=Gibbs_MuAlpha(ID,Z,X,T,Sigma,SigAlpha)
    %This function calculates updated Mu and Alpha (1D)
    
    if length(X)==1
        Mu=Gibbs_Mu(ID,Z,X,Sigma);
        Alpha = 0;
        return;
    end
    
    if sum(Z==ID)==0
        Mu = X(randi(length(X)));
        Alpha = -SigAlpha+2*SigAlpha*rand();
    else
        %Get the localizations from the IDth emitter
        Xs=X(Z==ID);
        Sigs = Sigma(Z==ID);
        Ts=T(Z==ID);

        A = sum(Sigs.^-2);
        B = sum(Ts./Sigs.^2);
        D = sum((Ts.^2)./(Sigs.^2));

        %MLE estimates of Mu and Alpha

        [Alpha,Center] = calAlpha(Xs,Sigs,Ts,SigAlpha);
        MA=[Center;Alpha];

        %Covariance matrix Sigma
        COV = pinv([A, B;B,D+1/SigAlpha^2]);

        %This draws [Mu,Alpha] from a multivariate normal
        MuAlpha=mvnrnd(MA,COV);
        Mu=MuAlpha(1);
        Alpha=MuAlpha(2);

        if Mu == Center
           Mu = Center + sqrt(A)*randn(); 
        end
    end
    
end

function [Mu]=Gibbs_Mu(N,Z,X,Sigma)
    % Sample a new emitter location along one axis 
    % for all allocated emitters in Z.
    Sigma_inv_sq = Sigma.^-2;
    
    A = accumarray(Z, X .* Sigma_inv_sq, [N 1]);  % Sum X / se^2 for all emitters
    B = accumarray(Z, Sigma_inv_sq, [N 1]);  % Sum se^-2 for all emitters

    XMLE = A./B;

    Mu = randn(size(B), 'like', B) ./ sqrt(B) + XMLE;
    Mu = Mu';
end

function [Alpha,Center] = calAlpha(Xs,Sigs,Frames,SigAlpha)
    Frames = single(Frames);
    A = sum(Xs./Sigs.^2);
    B = sum(Frames./Sigs.^2);
    C = sum(Sigs.^-2);
    AlphaTop = sum((C*Xs-A).*Frames./Sigs.^2);
    AlphaBottom = sum((C*Frames-B).*Frames./Sigs.^2)+C/SigAlpha^2;
    Alpha = AlphaTop/AlphaBottom;
    Center = (A-Alpha*B)/C;
end

function LogL = p_Alloc(pair_probs)
    % Calculate the log-likelihood of pair_probs representing the true
    % configuration of emitters.
    LogL = log(mean(pair_probs, 2));
    LogL = sum(LogL);
end

function P=computePairProbs(SMD, Mu_X, Mu_Y, Alpha_X, Alpha_Y)
    % Compute the probability that each localization (in SMD) belongs to
    % each emitter defined by Mu_X, Mu_Y, Alpha_X, Alpha_Y
    % Can be recalculated only when Mu or Alpha is changed.

    P = normpdf2d(SMD.X, SMD.Y, ...
        Mu_X + Alpha_X.*SMD.FrameNum, Mu_Y + Alpha_Y.*SMD.FrameNum, ...
        SMD.X_SE, SMD.Y_SE);
end

function P=normpdf2d(X, Y, Mu_X, Mu_Y, Sigma_X, Sigma_Y)
    % Normal PDF with a diagonal covariance matrix
    % Based on matlab's "mvnpdf" implementation, simplified for 2d

    quadform = ((X - Mu_X) ./ Sigma_X).^2 + ((Y - Mu_Y) ./ Sigma_Y).^2;

    P = exp(-0.5*quadform - log(2*pi)) ./ (Sigma_X .* Sigma_Y);
end
