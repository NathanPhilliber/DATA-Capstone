%Nmax=5 maximal number of resonances
%NmaxS=5; maximal number of shellmodes
%nc=10; number of channels
%K=1; number of columns
%scale, default to 1
%omegaShift=10;
%dG=1.8
%dGs=1.8

function [N, Dm, peakLocations, omega_res, NS, GammaAmp] = spectra_generator_v2(Nmax, NmaxS, nc, scale, omegaShift, dG, dGs, gammaAmpFactor, ampFactor, epsilon2)
rng('shuffle');
cnt=1;
K=1;
N=floor(rand*Nmax)+1;
NS=floor(rand*NmaxS)+1;


% omega=scale.*rand(K,N)+omegaShift;
% Gamma=scale./Nmax.*(1+dG.*(rand(1,N)-0.5))./4;
% omegaS=scale.*rand(1,NS)+omegaShift;%shellmodes location
% GammaS=scale./NmaxS.*(1+dGs.*(rand(1,NS)-0.5)).*4;%shellmodes width: of the scale of liquidmodes width multiplies by a factor >>1

omega=scale.*rand(1,N)+omegaShift;
GammaAmp=scale./(1+0.5*dG)./gammaAmpFactor;
% GammaAmp=scale./(1+0.5*dG)./4;
% GammaAmp=scale./(1+0.5*dG)./16;
% gammaAmpFactor is independent of data. It is a function of the zoom.
% dG does depend on the data. When dG becomes larger, it will be easier to read the data.
% Applying a model with diff. gammaAmpFactor doesn't make sense since this can be changed in preprocessing.
% Features will become smaller relative with increased resolution.
% Compare performance with spectral windows with same spectral density. Compare recall 2 peaks with recall 4 peaks.

Gamma=GammaAmp.*(1+dG*(rand(1,N)-0.5));
% Extreme values: [0.75, 1.25]
% dG should be varied and not GammaAmp since it's an arbitrary variable that we can manually change.
% dG --> within a 'zoomed' scale, how much variation?
% Compare performance of recall for # modes for one choice of Gamma with diff. number of modes with another Gamma.
% Ratio should be the same as the ratio of GammaAmp.
% Get best number of gammaAmpFactor.

omegaS=scale.*rand(1,NS)+omegaShift;%shellmodes location
GammaAmpS=GammaAmp.*10;
GammaS=GammaAmpS.*(1+dGs.*(rand(1,NS)-0.5));%shellmodes width: of the scale of liquidmodes width multiplies by a factor >>1

phase=zeros(1,N);
Amp=zeros(1,N);

phase0=2*pi*rand(nc.*K,N);
Amp0=rand(nc.*K,N);

phaseS=zeros(1,NS);%shellmodes phases
AmpS=zeros(1,NS);%shellmodes amplitudes

phase0S=2*pi*rand(nc.*K,NS);%shellmodes phases

% ASK ABOUT SHELLMODE AMPLITUDES.
Amp0S=rand(nc.*K,NS)./ampFactor;   %shellmodes amplitudes: the scale of the liquid modes divided by a factor >>1
%Amp0S=1.0/5.0;

%omega_res=1000 * omegaShift; % resolution in angular frequency domain
omega_res=floor(1/(GammaAmp.*(1-dG*0.5))).*50*(2*omegaShift+1); % resolution in angular frequency domain
Omegai=0;
Omegaf=2*omegaShift+1;
Omega=Omegai:(Omegaf-Omegai)/(omega_res-1):Omegaf;


%range=2500:3000;
%range=floor(omega_res*(1/2-1/2/omegaShift)):floor(omega_res*(1/2+1/2/omegaShift));range=floor(omega_res*(1/2-1/2/omegaShift)):floor(omega_res*(1/2+1/2/omegaShift));
%range = floor(omega_res * 1.0/2.0) : floor(omega_res*(1/2+1/2/omegaShift)); % range used to normalize spectrum amplitude
range=floor(omega_res*(1/2-1/2/(2*omegaShift+1))):floor(omega_res*(1/2+1/2/(2*omegaShift+1)));


M = length(range);

for k=1:K %number of subplots
    offset=0;
    for jj=(k-1).*nc+1:k.*nc
        L=zeros(1,omega_res);
        phase(:)=phase0(jj,:);
        Amp(:)=Amp0(jj,:);
        
        phaseS(:)=phase0S(jj,:);%shellmodes phases
        AmpS(:)=Amp0S(jj,:);%shellmodes amplitudes
        
        %         for i=1:N
        %             L=L+ Amp(i)/2*(exp(1i*phase(i))./(omega(i)+Omega+1i*Gamma(i))+exp(-1i*phase(i))./(Omega-omega(i)+1i*Gamma(i)));%analytical Fourier transform
        %         end
        %
        %
        %         for i=1:NS %adding shellmodes
        %          L=L+ AmpS(i)/2*(exp(1i*phaseS(i))./(omegaS(i)+Omega+1i*GammaS(i))+exp(-1i*phaseS(i))./(Omega-omegaS(i)+1i*GammaS(i)));%analytical Fourier transform
        %         end
        %
        %         cF=abs(L);
        %
        %         cF=cF+0.05.*max(cF).*rand(1,omega_res);%adding white noise
        %
        %         D=cF.^(2);
        
        
        
        epsilon=0.01;
        T=1.*log(1/epsilon)./(GammaAmp.*(1-dG*0.5));%truncation time;
        
        for i=1:N
            trunc=exp(-Gamma(i).*T).*exp(1i.*Omega.*T).*(omega(i).*sin(phase(i)+omega(i)*T)-(Gamma(i)-1i.*Omega).*cos(phase(i)+omega(i)*T));
            L=L+Amp(i).*(cos(phase(i)).*(Gamma(i)-1i.*Omega)-sin(phase(i)).*omega(i)+trunc)./(omega(i)^2+(Gamma(i)-1i.*Omega).^2);
        end
        for i=1:NS %adding shellmodes
            truncS=exp(-GammaS(i).*T).*exp(1i.*Omega.*T).*(omegaS(i).*sin(phaseS(i)+omegaS(i)*T)-(GammaS(i)-1i.*Omega).*cos(phaseS(i)+omegaS(i)*T));
            L=L+AmpS(i).*(cos(phaseS(i)).*(GammaS(i)-1i.*Omega)-sin(phaseS(i)).*omegaS(i)+0.*truncS)./(omegaS(i)^2+(GammaS(i)-1i.*Omega).^2);
        end
        
        cF=abs(L);
        D=cF.^(2);
        D=D + epsilon2.*max(D).*rand(1,omega_res);%adding white noise
        
        
        D=(D-min(D(range)))./(max(D(range))-min(D(range)));
        Dm(jj,:)=D(range);
        offsetm(jj)=offset;
        offset=offset+1;
        peakLocations=(omega-Omega(range(1)))./(Omega(range(end))-Omega(range(1)));
    end
    
end


