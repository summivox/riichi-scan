function [TH]=translada_CVhough_4(H1,H2,H3,H4, CsT, SnT, DeltaX,DeltaY, Ts, Smin, Smax, dS)
%
%  juntas as HT H1-H4 em uma imagem.
%
%
%tic;
s = size(H1);
TH = zeros(s(1), s(2));
    translacao1 = round((DeltaX*SnT+DeltaY*CsT)/dS);
    translacao2 = round((-DeltaX*SnT+DeltaY*CsT)/dS);
    translacao3 = round((DeltaX*SnT-DeltaY*CsT)/dS);
    translacao4 = round((-DeltaX*SnT-DeltaY*CsT)/dS);
ind_orig0=1:s(1);
for i=1:length(Ts);
    ind_new=ind_orig0-translacao1(i);
    ind_orig=ind_orig0(ind_new>0 & ind_new<=s(1));
    ind_new=ind_new(ind_new>0 & ind_new<=s(1));
    TH(ind_new,i) = TH(ind_new,i) + H1(ind_orig,i);

    ind_new=ind_orig0-translacao2(i);
    ind_orig=ind_orig0(ind_new>0 & ind_new<=s(1));
    ind_new=ind_new(ind_new>0 & ind_new<=s(1));
    TH(ind_new,i) = TH(ind_new,i) + H2(ind_orig,i);

    ind_new=ind_orig0-translacao3(i);
    ind_orig=ind_orig0(ind_new>0 & ind_new<=s(1));
    ind_new=ind_new(ind_new>0 & ind_new<=s(1));
    TH(ind_new,i) = TH(ind_new,i) + H3(ind_orig,i);
    
    ind_new=ind_orig0-translacao4(i);
    ind_orig=ind_orig0(ind_new>0 & ind_new<=s(1));
    ind_new=ind_new(ind_new>0 & ind_new<=s(1));
    TH(ind_new,i) = TH(ind_new,i) + H4(ind_orig,i);   
end;
%figure,imshow(TH1/5);figure,imshow(TH2/5);figure,imshow(TH3/5);figure,imshow(TH4/5);
%TH=TH1+TH2+TH3+TH4;
%toc;
