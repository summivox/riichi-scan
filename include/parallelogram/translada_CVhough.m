function [TH]=translada_CVhough(H, CsT, SnT, DeltaX,DeltaY, Ts, Smin, Smax, dS)

%tic;
s = size(H);
TH = zeros(s(1), s(2));

%Ts, Smin, Smax,
%DeltaX,



for i=1:size(Ts);
    
    for j=1:Smax-Smin,
        translacao = round((DeltaX*SnT(i)+DeltaY*CsT(i))/dS);
        indice = j - translacao;
        if ((indice>0)&(indice<s(1)+1))
            TH(indice,i) = H(j,i);
        end;
    end;
end;


%toc;
