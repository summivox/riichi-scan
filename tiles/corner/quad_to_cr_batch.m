function crs = quad_to_cr_batch(quads)

n = size(quads, 1)/4;
crs = zeros(n, 4);
for i = 1:n
    [c, r] = quad_to_cr(quads(i*4+(-3:0), :));
    crs(i, 1:2) = c;
    crs(i, 3:4) = r;
end

end
