function J = quad_to_rect(I, q, size)
% transform quadrangle patch on image to rectangle
%
% input:
%   I: image
%   q: quadrangle (4r2c)
%   size: [W H]
% output:
%   J: transformed rectangle image (H*W*?)

W = size(1);
H = size(2);

r = [1 1; W 1; 1 H; W H]; 
tf = fitgeotrans(q, r, 'projective');
J = imwarp(I, tf, 'OutputView', imref2d(size));

end

