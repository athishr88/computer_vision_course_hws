.function [outImg] = addRandomNoise_L(inImg)
%addRandomNoise adds random colours on random pixels of the image
%   Detailed explanation goes here

%generating random value between [-255 255] and adding it to input to get
%ouput
random_value = randi([-255, 255], size(inImg), 'uint8');
outImg = inImg + random_value;

%imwrite(f,"computervision/task5.png","png");
end

