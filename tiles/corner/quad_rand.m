function [aabb, pp] = quad_rand(p, padding_range)
% randomly select quadrangle and its padded axis-aligned bounding box
% 
% input:
%   p: groups of 4 rows of quadrangle coordinates
%   padding_range: [min max] of padding
% output:
%   aabb: [x1 y1 x2 y2]
%   pp: randomly selected quadrangle

n = size(p, 1)/4;
i = randi(n);
pp = p(i*4+(-3:0), :);

padding = randi(padding_range, 1, 2);
lo = min(pp) - padding;
hi = max(pp) + padding;
aabb = [lo hi];

end