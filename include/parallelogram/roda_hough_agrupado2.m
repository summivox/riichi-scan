function [H,rho, theta, pico, valor_pico]=roda_hough_agrupado2(H, rho, theta, RMin,pad);
%
% calcula a transformada hough para a imagem monocromatica x
% [H,berough,m,b]=hough_kittler(x,sT,dS,RMin)
% RMin e o menor rho utilizado
%show =1  %display lines
pico=[];
valor_pico=[];
nT=length(theta)-2*pad;
nS=length(rho);
%
%
%  CHAMAR A ROTINA DE EXTRAÇAO DE PICOS
%
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
%hh=round(.15*s(1));w=round(.1*s(2));
%hh=12;w=4; % tirado do paper furukawa
hh=9;w=round(.07*s(2));
mask=ones(hh,w)/hh/w;
be=H.^2./(conv2(H,mask,'same')+1e-10);
berough=be;
%
% mudancas
%
hh = ordfilt2(h,9,ones(3));
hh=wkeep(hh,[nS nT]);
be = ordfilt2(be,9,ones(3));
%berough=filter2(ones(3)/9,berough,'same');
%
% fim m udancas
%
berough=berough/(max(berough(:)+1e-10));
H=berough;
%
% desfaz o padding de CVHough_kittler_extended
%
H=wkeep(H,[nS nT]);
h=wkeep(h,[nS nT]);
theta=wkeep(theta,nT);
berough=wkeep(berough,[nS nT]);
be=wkeep(be,[nS nT]);
s=size(H);
mat_x=repmat((1:s(1))',[1 s(2)]);
mat_y=repmat(1:s(2),[s(1) 1]);
%disp(sprintf('valor maximo de H:%d ',max(H(:))));
%
% realiza non-maxima suppresion
%
Tn=.1;
szex=round(.1*s(1));szey=round(.1*s(2));
mx = ordfilt2(H,szex*szey,ones(szex,szey)); % Grey-scale dilate.
save teste h H be berough
H = (H==mx)&(H>Tn)&(hh>RMin)&(be>3*RMin);
%H1 = (H>Tn)&(h>RMin)&(be>5*RMin);
%figure,subplot(131);imshow(2*berough);subplot(132);imshow(h/20);subplot(133);imshow(H);
%
%
%  Pega apenas os Max_picos maiores picos
%
Max_picos=16;
lista=berough(H>0);
if ~isempty(lista),
    lista=sort(lista);
    lista=flipud(lista);
    thresh=lista(min(Max_picos,length(lista)));
    H(berough<thresh)=0;
    rho_ind=mat_x(H>0);
    theta_ind=mat_y(H>0);
    rho=rho(rho_ind);
    theta=theta(theta_ind);
    pico = [rho_ind, theta_ind];
    %valor_pico=h(H>0);
    for i=1:length(rho_ind),
        valor_pico=[valor_pico; sum(h(rho_ind(i)-1:rho_ind(i)+1,theta_ind(i)))];
    end,
else,
    h=[];
    valor_pico=[];
    theta=[];   
end,
%figure;subplot(121);imshow(H);subplot(122);imshow(h/5);
H=h;


