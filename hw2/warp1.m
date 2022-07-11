function [] = warp1(img1,img2, H)
% Function warps the two images on to a common mosaic and displays them

% loading the selected Homography matrix from .mat file
load('best_H.mat');

% inveres homography for reverse transforms
revH = inv(best_H);



% Bounding box creation
img2row = size(img2, 1);
img2col = size(img2, 2);

corners = [1 1 1; 1 img2col 1; img2row 1 1; img2row img2col 1]' ;
cornersR2L = round(right2Left(corners, revH));

allcorners = cat(2, corners(1:2, :), cornersR2L);
min_x = min(allcorners(1, :));
max_x = max(allcorners(1, :));

min_y = min(allcorners(2, :));
max_y = max(allcorners(2, :));

numRows = round(1.1*(max_x - min_x));
numCols = round(1.1*(max_y - min_y));

mosaic = zeros(numRows, numCols, 3)-1;


% Pasting Left Image
for row=1:size(img1, 1)
    for col=1:size(img1, 2)
        for depth=1:3
            mosaic(row-min_x+1, col-min_y+1, depth) = img1(row, col, depth);
        end
    end
end

% Pasting right image
for row=1:size(mosaic, 1)
    for col=1:size(mosaic, 2)
        for depth=1:3
            if mosaic(row, col, depth) == -1
                mosVec = [row, col];
                imgVec = [row+min_x-1 col+min_y-1 1]';
                rightRef = left2Right(imgVec, H);
                if  1 < rightRef(1)< size(img2, 1) && 1 < rightRef(2) < size(img2, 2)
                    mosaic(row, col, depth) = binomial(rightRef, img2, depth);
                end
                
            end
        end
    end
end

% convert the image to data type uint8 and display
mosaic = uint8(mosaic);
subplot(2, 2, 1), imshow(img1);
subplot(2, 2, 2), imshow(img2);
subplot(2, 2, 3:4), imshow(mosaic);


end

