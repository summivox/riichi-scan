function [x, y] = para_plot_one(p, varargin)
% plot one parallelogram (3p)

q = [p(1, :); p(2, :); para_3to1(p(1, :), p(2, :), p(3, :)); p(3, :); p(1, :)];
x = q(:, 1);
y = q(:, 2);
plot(x, y, varargin{:})

end