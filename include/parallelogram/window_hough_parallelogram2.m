function [img_out, retangulos, centros] = window_hough_parallelogram2(img,angulo)
%indice=zeros(size(img));
PONTOS=[];
centros=[];
RMin=5;
s=size(img);
RMax=ceil(s(1)/4);
%dT=.56RMax; % discretizacao no angulo
%dS=1/sqrt(2); % discretizacao na distancia
dT=3*pi/4/(2*RMax+1);
dS=3/4;
tic  % inicia o contador de tempo

TAM = 3;
[H_img, W_img] = size(img);

index = 0;

% 
% calculo do mapa de bordas
%
bw = edge(img, 'canny',[.1 .2],1);figure,imshow(bw);
[H,berough, rho, theta, pico, valor_pico]=roda_hough_parallelogram2(bw,dT,dS);
if ~isempty(H),
    PONTOS = find_parallelogram2(img, dT, H, rho, theta, pico, valor_pico, angulo);
%     if ~isempty(PONTOS),
%         retangulos(:,:,count)=PONTOS;
%         simetry(count,:)=simetria;
%         centros(count,:)=edgedata(:,n)';
%         indice(edgedata(1,n),edgedata(2,n))=count;
%         count=count+1;
%     end,
end,
img_out = H;
%if ~isempty(retangulos),
%    [retangulos,centros]=remove_duplicated_rectangles2(img, retangulos, centros, indice, simetry);
%end,
t=toc;  %marca o final do contador de tempo
minutos=floor(t/60);
segundos=(t-minutos*60);
disp(sprintf('tempo decorrido: %2d minutos e %2d segundos',minutos,round(segundos)))
