function PONTOS = find_parallelogram2(img_original, dT, rho, theta, pic, angulo)

%ret = find_simetric_lines(img, H, berough, rho, theta, nT, nS, b, m)
% rho = rho's retornados pelo kittler
% theta = theta's retornados pelo kittler
% dT eh o espacamento dos angulos theta
% pic eh a altura dos picos
% ret = boolean.  1 se encontrou uma estrutura

MIN_internal=20; %angulo interno minimo
ret = 0;
rect=[];
simetria=[];
% 
%   procura paralelogramos se ha mais do que 3 picos 
%
%
%rho_and_theta = [rho ; 180/pi*theta';pic'],%
if length(theta)>3,
%     if abs(theta(1))<4,
%         theta=[theta; -theta(1)];
%         rho=[rho rho(1)];
%         pic=[pic; pic(1)];
     end,
    ang = 180/pi*theta';
    L_theta_intra = max(5.1,3.1*180*dT/pi);  %limiar do angulo para o mesmo par
    L_theta_inter = max(5.1,3.1*180*dT/pi);   %limiar do angulo entre dois pares
    L_pico = 4; %limiar de tamanho dos lados dos retangulos (em %)
    %    L_theta_intra = 3;   %limiar do angulo para o mesmo par
    %    L_theta_inter = 3;   %limiar do angulo entre dois pares
    L_comp=.4;
    new_pic = []; 
    new_rho = [];
    new_rho1 = [];
    new_rho2 = [];
    new_theta = [];
    cont = 0;
    %
    %  procura por pares de picos com mesmo theta, e aproximadamente mesma altura de pico
    %
    for index1=1:length(rho);
        for index2=(index1+1):length(rho);
            diferenca_theta = abs(ang(index1)-ang(index2));
            if (diferenca_theta<=L_theta_intra);
                diferenca_pico=abs(pic(index1)-pic(index2));
                if (diferenca_pico<=(pic(index1)+ pic(index2))*L_pico/200),
                    %                if (diferenca_pico<=min(pic(index1),pic(index2))*L_pico/100),
                    cont = cont+1;
                    %
                    % pega a m´edia dos angulos e das distancias
                    %
                    new_theta(cont) = (theta(index1)+theta(index2))/2;
                    theta1(cont)=theta(index1);
                    theta2(cont)=theta(index2);
                    new_rho1(cont) = rho(index1);
                    new_rho2(cont)= rho(index2);
                    new_pic(cont)=(pic(index1)+pic(index2))/2;
                    %
                    % armazena as diferencas de angulos e distancias
                    %
                    dtheta(cont)=diferenca_theta;
                    drho(cont)=abs(rho(index1)-rho(index2));
                end
            end
        end
    end;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rho1 = new_rho1;
    rho2 = new_rho2;
    theta = new_theta;
    pic=new_pic;
    %[180/pi*theta; rho1; rho2; pic],%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ang = 180/pi*theta';
    indice = 0;
    %
    %   verifica se eh paralelogramo (ou seja, busca pares de picos com \delta\theta=angulo) 
    %
    for i_ang=1:length(angulo);
        if (length(ang)>1)
            for n1=1:length(ang),
                for n2=(n1+1):length(ang),
                    diferenca_ang = ang(n2)-ang(n1);min(diferenca_ang,180-diferenca_ang); %
                    erro=[drho(n1)/sin(angulo(i_ang)*pi/180), drho(n2)/sin(angulo(i_ang)*pi/180); pic(n2) pic(n1)];  
                    erro=abs(diff(erro)./(erro(1,:)+1e-10));
                    if (diferenca_ang>=MIN_internal) & (abs(diferenca_ang-angulo(i_ang))<=L_theta_inter | abs(diferenca_ang+angulo(i_ang)-180)<=L_theta_inter) & max(erro)<=L_comp,
                        %                 if abs(diferenca_ang-angulo(i_ang))<=L_theta_inter,
                        indice = indice+1;
                        %rect(:,:,indice)=[theta(n1)+diferenca_ang*pi/360 theta(n1)+diferenca_ang*pi/360 theta(n2)-diferenca_ang*pi/360 theta(n2)-diferenca_ang*pi/360;rho1(n1) rho2(n1) rho1(n2) rho2(n2) ] % armazena rhos e thetas pros paralelogramos
                        rect(:,:,indice)=[theta1(n1) theta2(n1) theta1(n2) theta2(n2);rho1(n1) rho2(n1) rho1(n2) rho2(n2) ]; % armazena rhos e thetas pros paralelogramos
                    end
                end;
            end;
        end;
    end,  
    
end,
PONTOS = [];

%
% plota o centro da regiao de busca
%


ss=size(img_original);
centrox=ceil(ss(1)/2);
centroy=ceil(ss(2)/2);

%
% acha as intersecçoes das retas 
%

%figure,imshow(img_original);hold on
hold on;
if ~isempty(rect),
    ss=size(rect);
    ss=[ss 1];
    for i=1:ss(3),
        %  pico,
        ret = 1;
        disp('Candidato a Paralelogramo detectado: ');
        theta=[rect(1,1,i) rect(1,2,i)];theta=[theta rect(1,3,i) rect(1,4,i)];
        rho=[rect(2,1,i) rect(2,2,i)];rho=[rho rect(2,3,i) rect(2,4,i)];
        %        [180/pi*theta;rho],%
        cont = 1;
        A=[cos(theta(1)) sin(theta(1));cos(theta(3)) sin(theta(3))];b=[rho(1) rho(3)]';
        P=inv(A)*b;P=P+[centrox centroy]';P1=P;
        A=[cos(theta(2)) sin(theta(2));cos(theta(3)) sin(theta(3))];b=[rho(2) rho(3)]';
        P=inv(A)*b;P=P+[centrox centroy]';P2=P;
        A=[cos(theta(2)) sin(theta(2));cos(theta(4)) sin(theta(4))];b=[rho(2) rho(4)]';
        P=inv(A)*b;P=P+[centrox centroy]';P3=P;
        A=[cos(theta(1)) sin(theta(1));cos(theta(4)) sin(theta(4))];b=[rho(1) rho(4)]';
        P=inv(A)*b;P=P+[centrox centroy]';P4=P;
        x=[P1 P2 P3 P4 P1]; 
        PONTOS(:,:,i)=x;
    end,
end;
