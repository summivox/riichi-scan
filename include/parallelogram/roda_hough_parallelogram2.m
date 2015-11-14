function [H, berough, rho, theta, pico, valor_pico]=roda_hough_extended_parallelogram2(mag,dT,dS)
%
% calcula a transformada hough para a imagem monocromatica x
% [H,berough,m,b]=hough_kittler(x,sT,dS,RMin)
% RMin e o menor rho utilizado
%show =1  %display lines
show =0;
rho=[];
theta=[];
pico=[];
RMin=0;  %nao serve pra nada !!!
s=size(mag);
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
edgedata=[ix-(ceil(s(1)/2)) iy-(ceil(s(2)/2))]';
%edgedata=[ix iy]';

%
% verifica se ha bordas
%
if size(edgedata)>0;
    %
    % calculo da transformada Hough
    %
    
%    [H,theta,rho]=CVhough_extended(edgedata,dT,dS);H=double(H);
    [H,theta,rho] = split_hough2(mag,dT,dS);
%    tic
    nT=length(theta)-10;
    nS=length(rho);
    %
    % acha as possiveis retas
    %
    h=H;
    s=size(H);
    %
    %  tamanho da vizinhanca para realce da borboleta
    %
    %
%    hh=round(.15*s(1));w=round(.1*s(2));
    hh=13;w=round(.1*s(2));
%    hh=13;w=9;
    mask=ones(hh,w)/hh/w;
    be=H.^2./(conv2(H,mask,'same')+1e-10);
    berough=be;
    berough=berough/max(berough(:));
    H=berough;
    %
    % desfaz o padding de CVHough_kittler_extended
    %
    H=H(:,6:(6+nT-1));
    h=h(:,6:(6+nT-1));
    theta=theta(6:(6+nT-1));
    berough=berough(:,6:(6+nT-1));
    be=be(:,6:(6+nT-1));
    s=size(H);
    %
    % elimina os valores de H com abs(rho)<RMin 
    %
    %
    elim=1:nS;
    elim=elim(abs(rho)<RMin);
    H(elim,:)=0;
    be(elim,:)=0;
    berough(elim,:)=0;
    mat_x=repmat((1:s(1))',[1 s(2)]);
    mat_y=repmat(1:s(2),[s(1) 1]);
    %disp(sprintf('valor maximo de H:%d ',max(H(:))));
    %
    % realiza non-maxima suppresion
    %
%    Tn=mean2(berough(berough>.1));  %para janelas pequenas, aumenta-se o valor do limiar
    Tn=.1;
%    szex = 19; szey = 13;                  % tamanho da vizinhanca
%    szex=round(.2*s(1));szey=round(.1*s(2));
    szex=15;szey=round(.1*s(2));
    mx = ordfilt2(H,szex*szey,ones(szex,szey)); % Grey-scale dilate.
    H = (H==mx)&(H>Tn)&(h>.1*max(h(:)));
%     H = (H==mx)&(H>.15)&(h>RMin)&(be>3*RMin);    
%    figure,subplot(131);imshow(h/10);subplot(132);imshow(2*berough);subplot(133);imshow(H)
    %
    %
    %  Pega apenas os Max_picos maiores picos
    %
    Max_picos=16;
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
        valor_pico=h(H>0);
    else,
        h=[];
        valor_pico=[];
    end,
    %figure;subplot(121);imshow(H);subplot(122);imshow(h/5);
    
end; % do if!

%disp(sprintf('number of lines:%d ', length(theta)));
H=h;


