function [H, berough, rho, theta, pico, valor_pico]=roda_hough_tile2(mag,dT,dS,rho_min, rho_max, CsT, SnT, Ts, RMin,pad);
%
% calcula a transformada hough para a imagem monocromatica x
% [H,berough,m,b]=hough_kittler(x,sT,dS,RMin)
% RMin e o menor rho utilizado
%show =1  %display lines
show =0;
rho=(rho_min:rho_max)*dS;
theta=Ts;
pico=[];
valor_pico=[];
s=size(mag);
berough=[];
%
% calculo das coordenadas das bordas
%
mat_x=repmat((1:s(1))',[1 s(2)]);
mat_y=repmat(1:s(2),[s(1) 1]);
ix=mat_x(mag>0);
iy=mat_y(mag>0);
%
% tranlacao da origem do sistema de coordenadas para o centro da janela
%
%edgedata=[ix-(ceil(s(1)/2)) iy-(ceil(s(2)/2))]';
edgedata=[ix-(s(1)+1)/2 iy-(s(2)+1)/2]';
%edgedata=[ix iy]';
H0=[];
%h=zeros(rho_max-rho_min+1,length(Ts));
%
% verifica se ha bordas
%
if ~isempty(edgedata);
    %
    % calculo da transformada Hough
    %
    H=CVhough_extended3(edgedata,dT,dS, rho_min, rho_max, CsT, SnT, Ts);
 %   H=CVhough_extended3_kittler(edgedata,dT,dS, rho_min, rho_max, CsT, SnT, Ts);
    H0=H;
    %    figure,imshow(H/10);
    %    tic
    nT=length(theta)-2*pad;
    nS=length(rho);
    %
    % 
    %   SUGESTAO: CRIAR UMA ROTINA PARA EXTRAÇ~AO DE PICOS
    %
    %
    % acha as possiveis retas
    %
    h=H;
    s=size(H);
    %
    %  tamanho da vizinhanca para realce da borboleta
    %
    %
    %    hh=13;w=13;
   % hh=round(.15*s(1));w=round(.1*s(2));
  %  hh=12;w=4; % tirado do paper furukawa
    hh=9;w=round(.07*s(2));
    mask=ones(hh,w)/hh/w;
    be=H.^2./(conv2(H,mask,'same')+1e-10);
    berough=be;
    berough=berough/max(berough(:));
    H=berough;
    %
    % desfaz o padding de CVHough_kittler_extended
    %
    H=wkeep(H,[nS nT]);
    h=wkeep(h,[nS nT]);
    theta=wkeep(theta,nT);
    berough=wkeep(berough,[nS nT]);
    be=wkeep(be,[nS nT]);
%    H=H(:,6:(6+nT-1));
%    h=h(:,6:(6+nT-1));
%    theta=theta(6:(6+nT-1));
%    berough=berough(:,6:(6+nT-1));
%    be=be(:,6:(6+nT-1));
    s=size(H);
    mat_x=repmat((1:s(1))',[1 s(2)]);
    mat_y=repmat(1:s(2),[s(1) 1]);
    %disp(sprintf('valor maximo de H:%d ',max(H(:))));
    %
    % realiza non-maxima suppresion
    %
    %    Tn=mean2(berough(berough>.1));  %para janelas pequenas, aumenta-se o valor do limiar
    Tn=.1;
%    szex=round(.2*s(1));szey=round(.1*s(2));
    szex=round(.1*s(1));szey=round(.1*s(2));
    mx = ordfilt2(H,szex*szey,ones(szex,szey)); % Grey-scale dilate.
    save teste h H be
    H = (H==mx)&(H>Tn)&(h>RMin)&(be>3*RMin);
 %   H1 = (H>Tn)&(h>RMin)&(be>3*RMin);
    %     H = (H==mx)&(H>.15)&(h>RMin)&(be>3*RMin);    
%    figure,subplot(131);imshow(2*berough);subplot(132);imshow(h/20);subplot(133);imshow(H)
%    figure,subplot(121);imshow(h/20);subplot(122);imshow(H)
    %
    %
    %  Pega apenas os Max_picos maiores picos
    %
    Max_picos=10;
    lista=berough(H>0);
    %    lista=h(H>0);
    if ~isempty(lista),
        lista=sort(lista);
        lista=flipud(lista);
        thresh=lista(min(Max_picos,length(lista)));
        H(berough<thresh)=0;
        %        H(h<=thresh)=0;
        rho_ind=mat_x(H>0);
        theta_ind=mat_y(H>0);
        rho=rho(rho_ind);
        theta=theta(theta_ind);
        pico = [rho_ind, theta_ind];
       % valor_pico=h(H>0);
       for i=1:length(rho),
           valor_pico=[valor_pico; sum(h(rho_ind(i)-1:rho_ind(i)+1,theta_ind(i)))];
       end,
    else,
        h=[];
        valor_pico=[];
        theta=[];
        rho=[];
    end,
    %figure;subplot(121);imshow(H);subplot(122);imshow(h/5);
else,
    rho=[];
    theta=[];
end,
% do if!

%disp(sprintf('number of lines:%d ', length(theta)));
H=H0;
