function quads = cr_to_quad_batch(crs)

n = size(crs, 1);
quads = zeros(n*4, 2);
for i = 1:n
    quads(i*4+(-3:0), :) = cr_to_quad(crs(i, 1:2), crs(i, 3:4));
end

end