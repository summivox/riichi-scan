hold off;
for i = 1:size(P, 3)
    pp = squeeze(P(:, :, i));
    hold off; imshow(Ihedx);
    hold on; plot(pp(2, :), pp(1, :), 'r-'); hold off;
    pause;
end
hold off;