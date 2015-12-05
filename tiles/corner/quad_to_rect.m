function J = quad_to_rect(I, q, size)
% transform quadrangle patch on image to rectangle
%
% input:
%   I: image
%   q: quadrangle (4r2c)
%   size: [H W]
% output:
%   J: transformed rectangle image (H*W*?)

H = size(1);
W = size(2);

r = [1 1; W 1; 1 H; W H]; 
tf = fitgeotrans(q, r, 'projective');
J = imwarp(I, tf, 'OutputView', imref2d(size));

end

