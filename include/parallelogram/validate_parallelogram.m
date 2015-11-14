function bool=validate_parallelogram(x,mag,paralelo,Tperim,Tarea);
%
% bool=validate_parallelogram(x,mag,paralelo,Tperim,Tarea) retorna o valor 1 se os pontos
% contidos na matriz paralelo correspondem de fato a um paralelogramo.
% Para tal, uma metrica de homogeneidade na imagem original x eh calculada.
%
% Tperim eh um limiar de perimietro, e Tarea eh um limiar
% de homogeneidade de area
%
%
bool=logical(0);
bw=roipoly(x,round(paralelo(2,:)),round(paralelo(1,:)));
xx=double(x(imerode(bw,ones(3))));
perim=bwperim(bw);
edg=imdilate(perim,ones(5));
%figure,imshow(perim);
%figure,imshow(mag&edg);
%figure,imshow(edg);
%mean(x)
v=paralelo(1,:);
w=paralelo(2,:);
d=0;for i=1:4,d=d+sqrt((w(i+1)-w(i))^2+(v(i+1)-v(i))^2);end
erro1=abs(sum(mag(edg))-sum(perim(perim>0)))/max(sum(perim(perim>0)),sum(mag(edg)));
[abs(sum(mag(edg))) sum(perim(perim>0))]
erro2=std(xx);
if erro2<Tarea & erro1<Tperim,
    bool=logical(1);
end,
[bool erro2 erro1]