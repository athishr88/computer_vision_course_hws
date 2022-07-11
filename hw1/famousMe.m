function [outImg] = famousMe(inImg, meLogo)
% famousMe function adds the cropped image of a face to the top left corner
% of image of the users choice

%   Image of the face is first converted to greyscale
bw = luminance_L(meLogo);

% mask matrix initialized 
mask = zeros(size(bw), 'uint8');

% a binary of the image with threshold 200 would be created in mask
% variable
for i = 1: size(meLogo,1)
    for j = 1: size(meLogo, 2)
        if bw(i,j) < 200
            mask(i,j) = 255;
        end
    end
end

% all pixel values which are 0s in the mask would be made 0 in the coloured
% image of the face as well
for i = 1: size(meLogo,1)
    for j = 1: size(meLogo, 2)
        if mask(i,j) == 0
            meLogo(i,j,:) = 0;
        end
    end
end

% Image is scaled down or up based on the resolution of the primary image. 
imgs_proportion = size(meLogo, 2)/size(inImg,2);
shrink_factor = 1/(imgs_proportion * 4);
shrinked_logo = scaleNearest(meLogo, shrink_factor);
outImg = inImg;

% the scaled down and cropped face is now placed in then primary image
for i =1:size(shrinked_logo,1)
    for j = 1:size(shrinked_logo,2)
        for k = 1:3
            if shrinked_logo(i, j, k) ~= 0
                outImg(i, j, k) = shrinked_logo(i, j, k);
            end
        end
    end
end
end

