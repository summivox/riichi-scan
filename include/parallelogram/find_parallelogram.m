function y=find_parallelogram(x,berough,angulo,nT,nS,m,b)
%
% a partir do mapa de Hough realcado (berough), procura paralelogramos com
% angulo de abertura angulo
%
H=berough;
s=size(H);

[cc,bb]=imhist(H);cc=conv(cc,ones(1,3)/3);
bb=bb(cc>5*512/s(1));Tn=max(bb); 
s=size(H);
mat_x=repmat((1:s(1))',[1 s(2)]);
mat_y=repmat(1:s(2),[s(1) 1]);
%
% realiza non-maxima suppresion
%
%    Tn=mean2(berough(berough>.05));
szex = 9; szey =17;                  % tamanho da vizinhanca
mx = ordfilt2(H,szex*szey,ones(szex,szey)); % Grey-scale dilate.
H = (H==mx)&(H>Tn);    
rho=mat_x(H>0);
theta=mat_y(H>0);


%[cc,bb]=imhist(H);cc=conv(cc,ones(1,3)/3);
%bb=bb(cc>12*512/s(1));Tn=max(bb); 
%s=size(H);
%mat_x=repmat((1:s(1))',[1 s(2)]);
%mat_y=repmat(1:s(2),[s(1) 1]);
%disp(sprintf('valor maximo de H:%d ',max(H(:))));
%szex = 9; szey =17;                  % tamanho da vizinhanca
%mx = ordfilt2(H,szex*szey,ones(szex,szey)); % Grey-scale dilate.
%H = (H==mx)&(H>Tn);    
%rho=mat_x(H>0);
%theta=mat_y(H>0);
%
% MOSTRA os maximos locais
%
%z1=berough;
%z2=berough;
%z3=berough;
%for i=1:length(rho),
%    z2(rho(i),theta(i))=0;
%    z1(rho(i),theta(i))=1;
%    z3(rho(i),theta(i))=0;
%end,
%z(:,:,1)=z1;    z(:,:,2)=z2;    z(:,:,3)=z3;
%z=3*z;z(z>1)=1;
%figure,imshow(z);
rho=(rho-b)/m;
Ts=[-pi/2:pi/nT:pi/2-pi/nT];
theta=Ts(theta);
lista=[rho 180/pi*theta']'; %pega os maximos locais
y=180*theta/pi,y=[y -y(1)];
T1=2; %limiar angular para picos alinhados
dif=diff(y);dif=abs(dif)<T1; 
[bw,num]=bwlabel(dif);bw=imdilate(bw,[1 1]);%pega os pontos a direita
if dif(length(bw))>0, %corrige os problemas de theta=90 e theta = -90
    y(bw==num)=-y(bw==num);bw(bw==num)=1;
    num=num-1;
end,
y=y(1:(length(y)-1));
cluster=[];
for i=1:num, cluster(i)=mean(y(bw==i));end % calcula a media dos clusters de picos

%
%  acha todos os pares de pares de angulos candidatos a paralelogramos
%
%
indice=1:num;
T2=2;
count=[];
for i=1:num-1,
    b=cluster((i+1):num)-cluster(i);
    l=i+indice(abs(b-angulo)<T2); %T2 limiar angular 2
    if not(isempty(l)),
        count=[count;[i l]];
    end,
end,
%
% plota os candidatos
% 
theta_old=theta;
rho_old=rho;
scount=size(count);
img=x;
s=size(img);
for ii=1:scount(1),
    theta=theta_old(bw==count(ii,1)|bw==count(ii,2));
    rho=rho_old(bw==count(ii,1)|bw==count(ii,2));
    figure,imshow(img);hold on;
    for j=1:length(theta);
        theta1=theta(j);
        pho1=rho(j);
        if abs(theta1)>=pi/4,
            x=1:s(1);
            y=(pho1-x*cos(theta1))/sin(theta1);
            x=x(y>0 & y<=s(2));
            y=y(y>0 & y<=s(2));
        else,
            y=1:s(2);
            x=(pho1-y*sin(theta1))/cos(theta1);
            y=y(x>0 & x<=s(1));
            x=x(x>0 & x<=s(1));
        end,
        
        plot(y,x,'r');axis ij;hold on;
    end,
    hold off
end,



