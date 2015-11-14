
function Menor = get_menor_lado(img);

[xi, yi, c] = impixel(img);
x = abs(xi(1)-xi(2))
y = abs(yi(1)-yi(2))

N = major_number(x,y); 


Menor = N