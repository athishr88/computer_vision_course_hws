function [outImgVal] = binomial(cords,img,depth)
% binomial function calculates the values of a pixel of a pre-allocated
% mosaic with imputs - homography transformed coordiantes, reference image
% and depth of the pixel

% Create a 2x2 matrix with rows and colums the ceil and floor of the
% decimal styled coordinates

% edges and vertices of the image have 2x1 1x2 or 1x1 matrix and that will
% be ensured in the code below
left_x = max(1, floor(cords(1)));
right_x = min(size(img,1), ceil(cords(1)));
up_y = max(1, floor(cords(2)));
down_y = min(size(img, 2), ceil(cords(2)));

% Matrix creation
temp = img(left_x:right_x, up_y: down_y, :);

% Mean of all values in the matrix (Binomial approximation)
outImgVal = mean(temp(:, :, depth), 'all');

end

