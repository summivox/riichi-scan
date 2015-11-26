function quad_plot_all(p, varargin)

n = size(p, 1);
hold on;
for i = 1:4:n
    quad_plot_one(p(i:(i+3), :), varargin{:});
end
hold off;

end