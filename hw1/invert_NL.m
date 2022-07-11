function [outImg] = invert_NL(inImg)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%inverting all the pixel at the same time
outImg = 255 - inImg;
subplot(1,2,1), imshow(inImg);
subplot(1,2,2), imshow(outImg);

%imwrite(outImg,"computervision/task4.png","png");
end

