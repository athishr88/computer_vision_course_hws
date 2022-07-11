function [outImg] = luminance_L(inImg)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%using the formula given in PDF to convert to grayscale
outImg1 = .299 * inImg(:, :, 1);
outImg2 = .587 * inImg(:, :, 2);
outImg3 = .114* inImg(:, :, 3);

outImg = outImg1 + outImg2 + outImg3;

%imwrite(f,"computervision/task6.png","png");
end

