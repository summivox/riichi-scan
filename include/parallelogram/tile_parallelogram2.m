function [paralelo, alfa, lpicos,til] = tile_parallelogram2(img,s_tile,RMin,angulo)
%
% a imagem (de bordas) img eh subdividada em imagens com tamanho s_tile, e paralelogramos
% com angulo interno "angulo" sao procurados. RMin ´e o lado minimo a ser procurado
%
% se o parametro "angulo" nao for entrado, o algoritmo busca paralelogramos genericos
%
%

%
%
%  Parametros:
%
%
Tperim=.2;
Tarea=150;

if nargin==3,
    generico=1;
else,
    generico=0;
end,
PONTOS=[];
paralelo=[];
s=size(img);
alfa=[];
%
%  discretizacao dos angulos e theta para Hough 
%
lado=ceil(max(s_tile));
% Leung
%dT=1/2/lado;
%dS=1/2/sqrt(2);
%Furukawa
%dT=min(3*pi/180,3*pi/4/lado/3);
%dS=.75;%1/sqrt(2);
% Yuen
%dT=2/lado;
%dS=1;
% entre Yuen e Diagonal (Leung)
%dT=1/lado;
%dS=1/sqrt(2);
% Fen
dT=pi/2/lado;
dS=pi/4;
count=0;
%
%
% definicao dos angulos e rhos (padding de "pad" unidades `a esquerda e direita)
%
%
pad=5;
Ts=[(-pi/2-dT*pad):dT:pi/2+(pad-1)*dT]';
rho_min = round(-norm(2*s_tile)/dS/2);  
rho_max = round(norm(2*s_tile)/dS/2);   
rho=(rho_min:rho_max)*dS;
%cos and sin of all the angles
CsT=cos(Ts);
SnT=sin(Ts);

tic  % inicia o contador de tempo
% 
% calculo do mapa de bordas
%
bw=img;
bw_orig=bw;
%
%  faz padding de bw, para ter numero inteiro de tiles
%
nbw=zeros(ceil(s(1)/s_tile(1))*s_tile(1),ceil(s(2)/s_tile(2))*s_tile(2));
nbw(1:s(1),1:s(2))=bw;bw=nbw;clear nbw
%   ra=rand(size(bw));
%   bwadd=ra>.985;bwremove=ra<.45;
%   disp(sprintf('bordas originais %d',sum(sum(bw))));
%   disp(sprintf('bordas adicionadas %d',sum(sum(bwadd))-sum(sum(bwadd&bw))));
%   disp(sprintf('bordas removidas %d',sum(sum(bwremove&bw))));
%   bw(bwadd)=1;bw(bwremove)=0;
%bw=imnoise(bw,'salt & pepper',.05);
%   bw_orig=bw(1:s(1),1:s(2));
%  save bordas_ruido bw_orig
%figure,imshow(bw);
%
%  mostra o grid no mapa de bordas
%
s=size(bw);
grid(:,:,1)=double(~bw);grid(:,:,2)=double(~bw);grid(:,:,3)=double(~bw);
%grid(:,:,1)=double(img)/255;grid(:,:,2)=double(img)/255;grid(:,:,3)=double(img)/255;
for i=s_tile(1):s_tile(1):s(1)-1;
    grid(i,1:s(2),1)=1;grid(i,1:s(2),2)=0;grid(i,1:s(2),3)=0;
end,
for j=s_tile(2):s_tile(2):s(2)-1;
    grid(1:s(1),j,1)=1;grid(1:s(1),j,2)=0;grid(1:s(1),j,3)=0;
end,
figure,imshow(grid);hold on;
%figure,imshow(img);hold on
%
% para cada "tile" da imagem, busca paralelogramos
%
s=size(img);
N_lin=ceil(s(1)/s_tile(1));  %numero de tiles
N_col=ceil(s(2)/s_tile(2));
%
%  array das transformadas Hough nos tiles
%
HH=zeros(length(rho),length(Ts),N_lin,N_col);
for i=1:N_lin;
    for j=1:N_col;
        [i j]
        WIN=imcrop(bw,[1+(j-1)*s_tile(2),1+(i-1)*s_tile(1),s_tile(2)-1,s_tile(1)-1]);
        [H,berough, rho_new, theta_new, pico, valor_pico]=roda_hough_tile2(WIN,dT,dS,rho_min, rho_max, CsT, SnT, Ts, RMin,pad);
        til{i,j}=H;
%        til{i,j}=[theta_new'*180/pi;rho_new;valor_pico'];
        %                    wins{i,j}=WIN;
        if(~isempty(H)),
            HH(:,:,i,j)=H;
            if generico==0,
                PONTOS = find_parallelogram2(WIN, dT, rho_new, theta_new, valor_pico, angulo);
                alph=angulo*ones(1,10);
            else,
                [PONTOS,alph] = find_parallelogram3(WIN, dT, rho_new, theta_new, valor_pico);
            end,
            if ~isempty(PONTOS),
                ss=size(PONTOS);ss=[ss 1];
                for k=1:ss(3);
                    %
                    %  translada o paralelogramo detectado para coordenadas globais 
                    %
                    PONTOS(2,:,k)=PONTOS(2,:,k)+(j-1)*s_tile(2);
                    PONTOS(1,:,k)=PONTOS(1,:,k)+(i-1)*s_tile(1);
                    %
                    %  verifica se o paralelogramo detectado esta dentro do tile
                    %
                    if (max(PONTOS(1,:,k))<=(i+.1)*s_tile(1))&(min(PONTOS(1,:,k))>=(i-1.1)*s_tile(1))&(max(PONTOS(2,:,k))<=(j+.1)*s_tile(2))&(min(PONTOS(2,:,k))>=(j-1.1)*s_tile(2)),
                        disp('paralelogramo detectado');
                        line(PONTOS(2,:,k),PONTOS(1,:,k),'LineWidth',2,'Color','r'); axis ij; hold on;
                        %
                        % verifica se o paralelogramo eh valido
                        %
                        parcand=[PONTOS(1,:,k);PONTOS(2,:,k)];
                        bool=validate_parallelogram(img,bw_orig,parcand,Tperim,Tarea);
                        if bool,
                            disp('paralelogramo validado');
                            %                plot(PONTOS(2,:),PONTOS(1,:),'x');axis ij; hold on;
                            line(PONTOS(2,:,k),PONTOS(1,:,k),'LineWidth',2,'Color','g'); axis ij; hold on;
                            count=count+1;
                            %
                            %  armazena os pontos na variavel paralelo
                            % 
                            paralelo(:,:,count)=PONTOS(:,:,k);
                            alfa(count)=alph(k);
                        end,
                    end,
                end,
            end,
        end,
    end,
end
size(HH)
%
%  agrupa regioes de 4 tiles, e busca paralelogramos nesses grupos
%
%
%        DeltaX = (s_tile(2)+0.5)/2;
%        DeltaY = (s_tile(1)+0.5)/2;   ERRADO
DeltaX = (s_tile(2)-1)/2;
DeltaY = (s_tile(1)-1)/2;
for i=1:N_lin-1,
    for j=1:N_col-1;
        H = translada_CVhough_4(HH(:,:,i,j),HH(:,:,i,j+1),HH(:,:,i+1,j),HH(:,:,i+1,j+1), CsT, SnT, DeltaX,DeltaY, Ts, rho_min, rho_max, dS);
        %        if i==2 & j==4, save verifica H;end
        %lpicos{i,j}=H;
        til{i,j}=H;
        [H,rho_new, theta_new, pico, valor_pico]=roda_hough_agrupado2(H, rho, Ts, RMin,pad);
           lpicos{i,j}=[theta_new'*180/pi;rho_new;valor_pico'];
        if ~isempty(H),
            if generico==0,
                PONTOS = find_parallelogram2(zeros(2*s_tile), dT, rho_new, theta_new, valor_pico, angulo);
                alph=angulo*ones(1,10);
            else,
                [PONTOS,alph] = find_parallelogram3(zeros(2*s_tile), dT, rho_new, theta_new, valor_pico);
            end,
            if ~isempty(PONTOS),
                ss=size(PONTOS);ss=[ss 1];
                for k=1:ss(3);
                    %                
                    %   translada os paralelogramos detectados para coordenadas globais
                    %                
                    PONTOS(2,:,k)=PONTOS(2,:,k)+(j-1)*s_tile(2);
                    PONTOS(1,:,k)=PONTOS(1,:,k)+(i-1)*s_tile(1);
                    %
                    %  verifica se o paralelogramo detectado esta dentro do block (com tolerancia de 10%)
                    %
                    if (max(PONTOS(1,:,k))<=1+(i+1.2)*s_tile(1))&(min(PONTOS(1,:,k))>=1+(i-1.2)*s_tile(1))&(max(PONTOS(2,:,k))<=1+(j+1.2)*s_tile(2))&(min(PONTOS(2,:,k))>=1+(j-1.2)*s_tile(2))
                        disp('paralelogramo detectado'); 
                        line(PONTOS(2,:,k),PONTOS(1,:,k),'LineWidth',2,'Color','r'); axis ij; hold on;
                        %
                        % verifica se o paralelogramo eh valido
                        %
                        parcand=[PONTOS(1,:,k);PONTOS(2,:,k)];
                        bool=validate_parallelogram(img,bw_orig,parcand,Tperim,Tarea);
                        if bool==1,
                            disp('paralelogramo validado');
                            %                plot(PONTOS(2,:),PONTOS(1,:),'x');axis ij; hold on;
                            line(PONTOS(2,:,k),PONTOS(1,:,k),'LineWidth',2,'Color','g'); axis ij; hold on;
                            %               
                            %                armazena os pontos na variavel paralelo
                            %               
                            count=count+1;
                            paralelo(:,:,count)=PONTOS(:,:,k);
                            alfa(count)=alph(k);    
                        end,
                    end,
                end,
            end,
        end, 
    end,
end,
%figure,imshow(H/10);

save verifica H HH rho_new theta_new pico valor_pico
clear H HH
%img_out = H;
t=toc;  %marca o final do contador de tempo
minutos=floor(t/60);
segundos=(t-minutos*60);
disp(sprintf('tempo decorrido: %2d minutos e %2d segundos',minutos,round(segundos)))
hold off;
alfa
figure,imshow(img);
ss=size(paralelo);
ss=[ss 1];
if ~isempty(paralelo);
    for k=1:ss(3);
        %    xx=[0 121];
        %    for p=1:4,
        %        yy=(paralelo(1,p+1,k)-paralelo(1,p,k))/(paralelo(2,p+1,k)-paralelo(2,p,k))*(xx-paralelo(2,p,k))+paralelo(1,p,k);
        %        line(xx,yy,'LineWidth',2,'Color','r'); axis ij; hold on;
        %    end,
        line(paralelo(2,:,k),paralelo(1,:,k),'LineWidth',3,'Color',[.55 .55 .55]); axis ij; hold on;
   %     line(paralelo(2,:,k),paralelo(1,:,k),'LineWidth',2,'Color','r'); axis ij; hold on;  
    end,
end
