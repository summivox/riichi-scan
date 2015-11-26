function quad_plot_one(p, varargin)

q = p([1 2 4 3 1], :);
x = q(:, 1);
y = q(:, 2);
plot(x, y, varargin{:})

end