function [J, q] = quad_rand_rot(I, p, k_range, a_range)
% randomly select a quadrangle corresponding to a tile in image, then crop
% a random square patch around it
% 
% input:
%   I: source image
%   p: groups of 4 rows of quadrangle coordinates
%   k_range: [min max] of padding, relative to tile size
%     e.g. [0.1 0.2] => extra 10% to 20% of tile size
%   a_range: [min max] of rotation (in radian)
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

% defaults
if ~exist('k_range', 'var') || isempty('k_range')
    k_range = [0.0 0.25];
end
if ~exist('a_range', 'var') || isempty('a_range')
    a_range = [-20 +20]*(pi/180);
end

% random params
k = rand*(k_range(2) - k_range(1)) + k_range(1);
a = rand*(a_range(2) - a_range(1)) + a_range(1);

% NOTE: transformations are (row vector * matrix transposed) according to
% MatLab conventions

% apply rotation only and calc AABB
rot = [cos(a) sin(a); -sin(a) cos(a)]; 
pp_rot = pp*rot;
pp_rot_1 = min(pp_rot);
pp_rot_2 = max(pp_rot);
x1 = pp_rot_1(1); y1 = pp_rot_1(2);
x2 = pp_rot_2(1); y2 = pp_rot_2(2);
w = x2 - x1; h = y2 - y1;

% calculate size and position of square patch
s = ceil(max(w, h)*(1+k));
x0 = x1 - rand*(s-w);
y0 = y1 - rand*(s-h);
trans = [-x0+1 -y0+1];

% final transformation
tf = affine2d([rot [0;0]; trans 1]);
J = imwarp(I, tf, 'OutputView', imref2d([s s]));
q = bsxfun(@plus, pp_rot, trans);

end
