function p = get_data_cursor(fig)
% list all data cursors in a figure in order of creation
% p: rows of [x, y]

dcm_obj = datacursormode(fig);
c_info = getCursorInfo(dcm_obj);
p = cell2mat({c_info.Position}.');
p = flipud(p);

end

