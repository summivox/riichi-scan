function H = split_hough(bw);

dS = .5;
dT = .02

s = size(bw);
rho_min = round(-norm(s)/2);
rho_max = round(norm(s)/2);

%rho_min = -70;
%rho_max = 70;


DeltaX = s(2)/2;
DeltaY = s(1)/2;



%defining the range of the orientations of line
Ts=[(-pi/2-dT*5):dT:pi/2+4*dT]';


%cos and sin of all the angles
CsT=cos(Ts);
SnT=sin(Ts);



bw1 = bw(1:s(1)/2, 1:s(2)/2);
bw2 = bw(1:s(1)/2, 1+s(2)/2:s(2));
bw3 = bw(1+s(1)/2:s(1), 1:s(2)/2);
bw4 = bw(1+s(1)/2:s(1), 1+s(2)/2:s(2));


figure, imshow(bw1);
figure, imshow(bw2);
figure, imshow(bw3);
figure, imshow(bw4);


data1 = get_edgedata(bw1);
data2 = get_edgedata(bw2);
data3 = get_edgedata(bw3);
data4 = get_edgedata(bw4);


[h1]=CVhough_extended3(data1,dT,dS, rho_min, rho_max, CsT, SnT, Ts); figure, imshow(h1/5);
disp('h1');size(h1)
[h2]=CVhough_extended3(data2,dT,dS, rho_min, rho_max, CsT, SnT, Ts); figure, imshow(h2/5);
disp('h2'); size(h2)
[h3]=CVhough_extended3(data3,dT,dS, rho_min, rho_max, CsT, SnT, Ts); figure, imshow(h3/5);
disp('h3'); size(h3)
[h4]=CVhough_extended3(data4,dT,dS, rho_min, rho_max, CsT, SnT, Ts); figure, imshow(h4/5);
disp('h4'); size(h4)

[th1]=translada_CVhough(h1, CsT, SnT, DeltaX/2, DeltaY/2, Ts, rho_min, rho_max, dS); figure, imshow(th1/5);
disp('th1'); size(th1)
[th2]=translada_CVhough(h2, CsT, SnT, -DeltaX/2, DeltaY/2, Ts, rho_min, rho_max, dS); figure, imshow(th2/5);
disp('th2'); size(th2)
[th3]=translada_CVhough(h3, CsT, SnT, DeltaX/2, -DeltaY/2, Ts, rho_min, rho_max, dS); figure, imshow(th3/5);
disp('th3'); size(th3)
[th4]=translada_CVhough(h4, CsT, SnT, -DeltaX/2,-DeltaY/2, Ts, rho_min, rho_max, dS); figure, imshow(th3/5);
disp('th3'); size(th3)


H = th4+th1+th2+th3;
figure, imshow(H/10);

data = get_edgedata(bw);
[teste]=CVhough_extended3(data,dT,dS, rho_min, rho_max, CsT, SnT, Ts); figure, imshow(teste/10);
