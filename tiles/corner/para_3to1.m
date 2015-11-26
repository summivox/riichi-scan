function BR = para_3to1(TL, TR, BL)
% parallelogram: given 3 points, get 4th point
% assuming that all points are vectors of same shape
BR = TR + BL - TL;
