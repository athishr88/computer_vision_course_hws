function [outImg] = makeBright_L(inImg,brightness)
%makeBight_L will brighten the inImg. Amount of brightness depends on the
%argument 'brightness'
%   Detailed explanation goes here
%   Each pixel values of the image will be brightened to a value of the
%   input variable 'brightness'
outImg = zeros(size(inImg), 'uint8');
%loops through every pixel to get the more brightnees basically increasing
%or decrreasing every pixel value

for i = 1:size(inImg, 1)
    for j = 1:size(inImg, 2)
        for k = 1:size(inImg, 3)
            outImg(i, j, k) = inImg(i, j, k) + brightness;
        end
    end
end


%imwrite(outImg,"computervision/task1.png","png");

end

