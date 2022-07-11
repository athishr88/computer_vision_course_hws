function [outImg] = binaryMask(inImg)
% binaryMask function creates a boolean image based on a threshold
% determined from the image

% check if the image is colored or greyscale
if size(inImg, 3) > 1
    inImg2 = luminance_L(inImg);
else
    inImg2 = inImg;
end

% the mean pixel value from the greyscale image is calculated
meanVal = mean(inImg2(:));

% initializing resulting image with all values 0
outImg = zeros(size(inImg2),'uint8');

% assigning a value of 255 to pixels which crossed the threshold value
for i = 1:size(inImg2,1)
    for j = 1:size(inImg2,2)
        if inImg(i,j)<meanVal
            outImg(i,j)=255;
            
            
        end
        
    end
    
end
end

