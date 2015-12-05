function q = cr_to_quad(center, radius)
% (center, radius) back to square (in quadrangle format)

u = radius; % pointing down
v = [u(2), -u(1)]; % pointing right

A = center - u - v;
B = center - u + v;
C = center + u - v;
D = center + u + v;
q = [A; B; C; D];

end