function [outImg] = makeBright_NL(inImg,brightness)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%increseing the brightness of every pixel, if its greater than 255 it makes
%it 255
outImg = uint8(inImg + brightness);
subplot(1,2,1), imshow(inImg);
subplot(1,2,2), imshow(outImg);

%imwrite(outImg,"computervision/task2.png","png");

end

