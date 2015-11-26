%% set path and import packages
init;

%% load photo
I = imread('path/to/photo.jpg');
figure(1); imdisp(I); % imshow also okay

%% (mark corners using data cursor tool)

%% collect data cursors into `p` and visualize on `figure(2)`
dc2p; 

%% (adjust some corners, again using data cursor tool)

%% refresh `p` and `figure(2)`
dc2p;

%% (... rinse and repeat ...)

%% save data
savejson('', p, 'Compact', 1, 'FileName', 'path/to/photo.json');