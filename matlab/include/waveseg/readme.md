# Color Image segmentation

http://www.inf.ufrgs.br/~crjung/software.htm
 
Usage:

```matlab
y = waveseg_color_prl07(x,J,Tc);
```

Here, x is the input color image (uint8 array with dimensions m x n x 3), J is the number of scales used in the wavelet decomposition (optional, default according to paper), Tc is the threshold used for region merging (optional, default = 10), and y is the piecewise constant segmented image
