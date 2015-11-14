function  edgedata = get_edgedata(img_edge);

mag=img_edge;
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

