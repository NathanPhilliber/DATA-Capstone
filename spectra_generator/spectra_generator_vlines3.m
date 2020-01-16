function spectra_generator
cnt=1;

while(1)
    
    datafile = 'MyTrainingN3K80b';
    max_resonances=5;%maximal number of resonances
    
    N=floor(rand*max_resonances)+1;
    num_channels=10;%number of channes in one column
    num_columns=3;%number of number of columns
    
    scale=1;
    omega_shift=10;
    omega=scale.*rand(1,N)+omega_shift;
    Gamma=scale./max_resonances.*(1+1.8.*(rand(1,N)-0.5))./4;
    
    
    1
    phase=zeros(1,N);
    Amp=zeros(1,N);
    
    
    phase0=2*pi*rand(num_channels.*num_columns,N);
    Amp0=rand(num_channels.*num_columns,N);

    n=1000*omega_shift;
    Omegai=0;
    Omegaf=2*omega_shift+1;
    Omega=Omegai:(Omegaf-Omegai)/(n-1):Omegaf;
    
    %range=2500:3000;
    range=floor(n*(1/2-1/2/omega_shift)):floor(n*(1/2+1/2/omega_shift));
    
    
    figure(131)
    subplot(1,1,1) %to reset the figure
    for k=1:num_columns %number of subplots
        offset=0;
        for jj=(k-1).*num_channels+1:k.*num_channels
            L=zeros(1,n);
            phase(:)=phase0(jj,:);
            Amp(:)=Amp0(jj,:);
            
            %n is something like 10,000 but Amp and phase are single digit
            %so indexing fails
            for i=1:N
                L=L+ Amp(i)/2*(exp(1i*phase(i))./(omega(i)+Omega+1i*Gamma(i))+exp(-1i*phase(i))./(Omega-omega(i)+1i*Gamma(i)));%analytical Fourier transform
                
            end
            
            cF=abs(L);
            D=cF.^(2);
            D=(D-min(D(range)))./(max(D(range))-min(D(range)));
            Dm(jj,:)=D(range);
            offsetm(jj)=offset;
            offset=offset+1;
        end
        
       
    end
    
    for k=1:num_columns %number of subplots
        NumModes(cnt,k)=0;
        loc=range(1);
        dl=floor(length(range).*scale/max_resonances/10);
        
        while(1)
            for kk=1:k
                offset=0;
            for jj=(kk-1).*num_channels+1:kk.*num_channels
                
                subplot(1,num_columns,kk)
                plot((Omega(range)-Omega(range(1)))./(Omega(range(end))-Omega(range(1))), Dm(jj,:)+offset,'k')
                offset=offset+1;
                hold on
                axis tight
            end
            line(([Omega(loc) Omega(loc)]-Omega(range(1)))./(Omega(range(end))-Omega(range(1))), [0 offset-1+max(Dm(jj,:))]);
            hold off
            
            end  
            
            reply = input('Enter  ''1'' for feature, enter to move forward, ''l'' to move backword, ''q'' to quit \n','s');
%             if isempty(reply)
%                 reply = '+';
%             end
            
            if strcmpi(reply,'q'), break;
            elseif isempty(reply)
                loc=loc+dl;
            elseif strcmpi(reply,'1')
                NumModes(cnt,k)   = NumModes(cnt,k)+1;  %#ok<SAGROW>
                loc=loc+dl;
            else
                
                loc=loc-dl;
            end

            %             for kk=1:k
            %             subplot(1,K,kk)
            %             hold off
            %             end

            
            if loc>range(end)
                break
            end
        end
            if strcmpi(reply,'q'), break;
            end
        NumModesTrue(cnt)=n;       %#ok<SAGROW>
        GammaAv(cnt)=mean(Gamma).*max_resonances;       %#ok<SAGROW>
        save(datafile, 'NumModes', 'NumModesTrue','GammaAv');
    end




    if strcmpi(reply,'q'), break;
    else
        cnt
        cnt = cnt + 1;
    end
    
    n
    Guess=max(NumModes(cnt-1,:))
    
            for kk=1:num_columns
                offset=0;
            for jj=(kk-1).*num_channels+1:kk.*num_channels
                
                subplot(1,num_columns,kk)
                plot((Omega(range)-Omega(range(1)))./(Omega(range(end))-Omega(range(1))), Dm(jj,:)+offset,'k')
                offset=offset+1;
                hold on
                axis tight
                
            end
                plot((omega-Omega(range(1)))./(Omega(range(end))-Omega(range(1))),0,'-bo')
            end 
             [omega,I]=sort(omega);
             omega
             Gamma=Gamma(I)
%     for k=1:K
%         subplot(1,K,k)
%         plot(omega,0,'-bo')
%         hold off
%     end




    reply = input('Confirm seeing the resonances (''enter'')\n', 's');
end

if cnt==1
    fprintf('No training data captured.\n');
    return;
end

%% Save out the data
%datafile = name; %compose("TrainingData %s",datestr(now));
%save(datafile, 'NumModes', 'NumModesTrue','GammaAv');

% figure(151)
% col='*';
% load 'MyTrainingN3K80'
% [L,K]=size(NumModes)
% 
% for k=1:K
%     frac=0;
% for i=1:L
%     if NumModes(i,k)==NumModesTrue(i)
%         frac=frac+1;
%     end
%     %     plot(i,NumModesTrue(i)-NumModes(i),'o')
%     %     hold on
%     %     plot(i,NumModesTrue(i),'*')
% end
% plot(10.*k,frac./L,col)%plotting the actual mean Gamma
% hold on
% end
% %axis([0 0.5 0 1])

function f=myfun(t, omega,Gamma,phase,Amp)
f=0;
N=length(omega);
for i=1:N
    f=f+Amp(i).*sin(omega(i).*t+phase(i)).*exp(-Gamma(i).*t);
end

% function table_out(omega,Gamma)
% N=size(omega);
% [omega,I]=sort(omega);
% Gamma=Gamma(I);
% Parameters = {'omega';'Gamma'};
% Modes = 1:N;
% Modes_location_and_width=table(Modes,'RowNames',Parameters)
% 
% function table_out(inputs,outputs)
% 
% Parameters = {'tau_c';'N_c';'Theta_0c';'I_tilda_a';'I_tilda_c';'r';'N_a';'tau_a';'Theta_0a'};
% best_fit = outputs';
% %p =p';
% upper_bound = inputs(2,:)';
% lower_bound=inputs(3,:)';
% 
% Parametric_Values_and_Constraints=table(best_fit,upper_bound,lower_bound,...
%     'RowNames',Parameters)
