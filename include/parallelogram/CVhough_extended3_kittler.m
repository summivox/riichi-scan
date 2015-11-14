function H=CVhough_extended3_kittler(edgedata,dT,dS, rho_min, rho_max, CsT, SnT, Ts)
%CVhough Hough transform of a binary matrix
%
%function [H,Ts,rho]=CVhough_extended2(edgedata,dT,dS)
%         edgedata a 2-row matrix, with the x and y coordinates of the edges
%         dT is the orientation(thetas) step 
%         dS is the distance step                                
%         H votes histogram (distances - vertical x angles - horizontal)
%         Ts is the orientations vector
%         rho is the distances vector

MAXDIST=1.2;

%tic;


if nargin<3
   error('wrong number of parameters')
end

if (~isempty(edgedata))


row=edgedata(1,:)';
col=edgedata(2,:)';




%solving for distances for all orientations at all nonzero pixels
%size of S is: [length(row) , length(Ts)]
S=row*CsT' + col*SnT';


%mapping:
%         Smin = min(S(:))--> 1
%         Smax = max(S(:))--> nS
%gives (y=mx+b):
%         m=(nS-1)/(Smax-Smin)
%         b=(Smax-nS*Smin)/(Smax-Smin)
%and then round it and get rounded mapped S:rmS


rmS=(S/dS);
Smin=round(min(rmS(:)));
Smax=round(max(rmS(:)));
%Smin  = rho_min;
%Smax = rho_max;


%m =(nS-1)/(Smax-Smin);
%b =(Smax-nS*Smin)/(Smax-Smin);
%rmS=round(m*S + b);




%Note: H is [nT,nS]
%                                 rmS is [nP,nT]  nP:number of edge points

H=zeros(rho_max-rho_min+1, length(Ts));
w=1;
for k=Smin:Smax,
    isEq=abs(rmS-k)/w;isEq(isEq>1)=0;
    H(k-rho_min+1,:)=sum((isEq>0)-2*isEq.^2+isEq.^4,1);
end
rho=(Smin:Smax)*dS;

else H=zeros(rho_max-rho_min+1, length(Ts));
end;

%toc;

