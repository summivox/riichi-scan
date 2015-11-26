function [J, q] = quad_rand_rot(I, p, k_range)
% randomly select a quadrangle corresponding to a tile in image, then crop
% a random square patch around it
% 
% input:
%   I: source image
%   p: groups of 4 rows of quadrangle coordinates
%   padding_range: [min max] of padding, relative to tile size
%     e.g. [0.1 0.2] => extra 10% to 20% of tile size
% output:
%   J: cropped square patch
%   q: coordinates of the quadrangle in J frame
% 
% example:
%   [J, q] = quad_rand_rot(I, p, [0.4 0.6]);
%   h = imdisp(J);
%   makedatatip(h, round(q));


% pick a tile
n = size(p, 1)/4;
i = randi(n);
pp = p(i*4+(-3:0), :);

% get tile (center, radius)
pc = mean(pp);
dp = bsxfun(@minus, pp, pc);
r = sqrt(max(sum(dp.*dp, 2)));

% make transformation
% NOTE: row vectors * transposed matrices (MatLab conventions)
k = rand*(k_range(2) - k_range(1)) + k_range(1);
alpha = rand*(2*pi);
rot = [cos(alpha) sin(alpha); -sin(alpha) cos(alpha)]; 
trans = -pc*rot + (1+k)*r*[1 1] + (rand(1, 2) - 0.5)*2*k*r;
tf = affine2d([rot [0;0]; trans 1]);

% transform both image and points
J = imwarp(I, tf, 'OutputView', imref2d(ceil(2*(1+k)*r*[1 1])));
q = tf.transformPointsForward(pp);

end
