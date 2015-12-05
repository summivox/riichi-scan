function [center, radius] = quad_to_cr(q)

A = q(1, :);
B = q(2, :);
C = q(3, :);
D = q(4, :);
center = mean(q);
dir = ((C-A)+(D-B))/4*1.2;
dir = dir./norm(dir);
len = max(abs(dir*bsxfun(@minus, q, center).'));
radius = len*dir;

end

