function spectra_generator
cnt=1;


Nmax=5;%maximal number of resonances
NmaxS=5; %maximal number of shellmodes

N=floor(rand*Nmax)+1;
NS=floor(rand*NmaxS)+1;% number of shellmodes 
nc=10;%number of channes in one column
K=1;%number of number of columns

scale=1;
omegaShift=10;

omega=scale.*rand(1,N)+omegaShift;
Gamma=scale./Nmax.*(1+1.8.*(rand(1,N)-0.5))./4; 
omegaS=scale.*rand(1,NS)+omegaShift;%shellmodes location
GammaS=scale./NmaxS.*(1+1.8.*(rand(1,NS)-0.5)).*4;%shellmodes width: of the scale of liquidmodes width multiplies by a factor >>1

phase=zeros(1,N);
Amp=zeros(1,N);

phase0=2*pi*rand(nc.*K,N);
Amp0=rand(nc.*K,N);    

phaseS=zeros(1,NS);%shellmodes phases
AmpS=zeros(1,NS);%shellmodes amplitudes

phase0S=2*pi*rand(nc.*K,NS);%shellmodes phases
Amp0S=rand(nc.*K,NS)./5;   %shellmodes amplitudes: the scale of the liquid modes divided by a factor >>1

n=1000*omegaShift;
Omegai=0;
Omegaf=2*omegaShift+1;
Omega=Omegai:(Omegaf-Omegai)/(n-1):Omegaf;

%range=2500:3000;
range=floor(n*(1/2-1/2/omegaShift)):floor(n*(1/2+1/2/omegaShift));


for k=1:K %number of subplots
    offset=0;
    for jj=(k-1).*nc+1:k.*nc
        L=zeros(1,n);
        phase(:)=phase0(jj,:);
        Amp(:)=Amp0(jj,:);

        phaseS(:)=phase0S(jj,:);%shellmodes phases
        AmpS(:)=Amp0S(jj,:);%shellmodes amplitudes

        for i=1:N
            L=L+ Amp(i)/2*(exp(1i*phase(i))./(omega(i)+Omega+1i*Gamma(i))+exp(-1i*phase(i))./(Omega-omega(i)+1i*Gamma(i)));%analytical Fourier transform
        end


        for i=1:NS %adding shellmodes
         L=L+ AmpS(i)/2*(exp(1i*phaseS(i))./(omegaS(i)+Omega+1i*GammaS(i))+exp(-1i*phaseS(i))./(Omega-omegaS(i)+1i*GammaS(i)));%analytical Fourier transform
        end

        cF=abs(L);

        cF=cF+0.1.*max(cF).*rand(1,n);%adding white noise

        D=cF.^(2);
        D=(D-min(D(range)))./(max(D(range))-min(D(range)));
        Dm(jj,:)=D(range);
        offsetm(jj)=offset;
        offset=offset+1;
    end
end


dlmwrite('data.csv',N,'delimiter',',');
dlmwrite('data.csv',Dm,'delimiter',',','-append');


