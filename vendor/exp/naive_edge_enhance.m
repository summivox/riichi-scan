filt_log = fspecial('log', 10, 2);
im1_log = imfilter(im1, filt_log);
im1_enh = im1 + im1_log*15;
im1_enh_med = medfilt2(rgb2gray(im1_enh), [3, 3]);
figure; imshow(im1_enh_med);