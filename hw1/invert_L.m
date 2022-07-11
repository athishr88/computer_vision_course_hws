function [outImg] = invert_L(inImg)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%taking the output image same as input image
outImg = zeros(size(inImg), 'uint8');
%looping through every pixel to invert it by 255-value of it 
for i = 1:size(inImg, 1)
    for j = 1:size(inImg, 2)
        for k = 1:size(inImg, 3)
            outImg(i,j,k) = 255 - inImg(i,j,k);
        end
    end
end
subplot(1,2,1), imshow(inImg);
subplot(1,2,2), imshow(outImg);

%imwrite(outImg,"computervision/task3.png","png");
end

