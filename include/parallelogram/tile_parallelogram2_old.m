function [paralelo, HH] = tile_parallelogram2(img,angulo,s_tile)
% [img_out, retangulos, centros] = tile_parallelogram2(img,angulo)
%
% a imagem img eh subdividada em imagens com tamanho s_tile, e paralelogramos
% com angulo interno "angulo" sao procurados.
%
paralelo=[];
s=size(img);
lado=ceil(max(s_tile));
dT=3*pi/4/lado/2;
dS=.75;
count=0;
%
%
% definicao dos angulos e rhos
%
%
Ts=[(-pi/2-dT*5):dT:pi/2+4*dT]';
%rho_min = round(-norm(s_tile)/dS/2);  % deve tirar o 2 dividindo para juntar quatro janelas
%rho_max = round(norm(s_tile)/dS/2);   % deve tirar o 2 dividindo para juntar quatro janelas
rho_min = round(-norm(2*s_tile)/dS/2);  % deve tirar o 2 dividindo para juntar quatro janelas
rho_max = round(norm(2*s_tile)/dS/2);   % deve tirar o 2 dividindo para juntar quatro janelas

rho=(rho_min:rho_max)*dS;
%cos and sin of all the angles
CsT=cos(Ts);
SnT=sin(Ts);

tic  % inicia o contador de tempo
% 
% calculo do mapa de bordas
%
bw = edge(img, 'canny',[.1 .2],1);
%figure,imshow(bw);
%
%  mostra o grid no mapa de bordas
%
grid(:,:,1)=double(bw);grid(:,:,2)=double(bw);grid(:,:,3)=double(bw);
for i=1:s_tile(1):s(1);
    grid(i,1:s(2),1)=1;grid(i,1:s(2),2)=0;grid(i,1:s(2),3)=0;
end,
for j=1:s_tile(2):s(2);
    grid(1:s(1),j,1)=1;grid(1:s(1),j,2)=0;grid(1:s(1),j,3)=0;
end,
figure,imshow(grid);
    


%
% para cada "tile" da imagem, busca paralelogramos
%
s=size(img);
N_lin=s(1)/s_tile(1);  %numero de tiles
N_col=s(2)/s_tile(2);
%
%  array das transformadas Hough nos tiles
%
HH=zeros(length(rho),length(Ts),N_lin,N_col);
for i=1:N_lin;
    for j=1:N_col;
        WIN=imcrop(bw,[1+(i-1)*s_tile(1),1+(j-1)*s_tile(2),s_tile(1)-1,s_tile(2)-1]);
%        WIN=imcrop(bw,[1+(j-1)*s_tile(2),1+(i-1)*s_tile(1),s_tile(2)-1,s_tile(1)-1]);
        [H,berough, rho_new, theta_new, pico, valor_pico]=roda_hough_tile2(WIN,dT,dS,rho_min, rho_max, CsT, SnT, Ts);
        if(~isempty(H)),
            HH(:,:,i,j)=H;
%                         figure,imshow(WIN),
%            figure,imshow(H/10);
%            theta_new
            PONTOS = find_parallelogram2(WIN, dT, H, rho_new, theta_new, pico, valor_pico, angulo);
            if ~isempty(PONTOS),
                %
                %  translada o paralelogramo detectado para coordenadas globais 
                %
                PONTOS(2,:)=PONTOS(2,:)+(i-1)*s_tile(1);
                PONTOS(1,:)=PONTOS(1,:)+(j-1)*s_tile(2);
                line(PONTOS(2,:),PONTOS(1,:),'LineWidth',2,'Color','g'); axis ij; hold on;
                count=count+1;
                %
                %  armazena os pontos na variavel paralelo
                % 
                paralelo(:,:,count)=PONTOS;
            end,
        end,
    end,
end
size(HH)
%
%  agrupa regioes de 4 tiles, e busca paralelogramos nesses grupos
%
%
for i=1:N_lin-1,
    for j=1:N_col-1;
        DeltaX = s_tile(2);
        DeltaY = s_tile(1);
        H = translada_CVhough_4(HH(:,:,i,j),HH(:,:,i+1,j),HH(:,:,i,j+1),HH(:,:,i+1,j+1), CsT, SnT, DeltaX/2,DeltaY/2, Ts, rho_min, rho_max, dS);
%        if i==2 & j==2,figure,imshow(H/10);pause;end
%        figure,imshow(H/10);
%if i==7 & j==3,
        [rho_new, theta_new, pico, valor_pico]=roda_hough_agrupado2(H, rho, Ts);
        %    end,
        if ~isempty(H),
            PONTOS = find_parallelogram2(zeros(2*s_tile), dT, H, rho_new, theta_new, pico, valor_pico, angulo);
            if ~isempty(PONTOS),
                %
                %  translada o paralelogramo detectado para coordenadas globais 
                %
                PONTOS(2,:)=PONTOS(2,:)+(i-1)*s_tile(1);
                PONTOS(1,:)=PONTOS(1,:)+(j-1)*s_tile(2);
                line(PONTOS(2,:),PONTOS(1,:),'LineWidth',2,'Color','g'); axis ij; hold on;
                count=count+1;
                %
                %  armazena os pontos na variavel paralelo
                % 
                paralelo(:,:,count)=PONTOS;
            end,
        end, 
    end,
    
end,
end,
%figure,imshow(H/10);


%img_out = H;
t=toc;  %marca o final do contador de tempo
minutos=floor(t/60);
segundos=(t-minutos*60);
disp(sprintf('tempo decorrido: %2d minutos e %2d segundos',minutos,round(segundos)))

