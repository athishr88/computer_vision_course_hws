function  [outImg]  = scaleBilinear(inImg,factor)

%scaleBilinear function scale up or down an image based on bilinear
%appoximation method

% Initializing the resulting image with resolution as per the scaling
% required
inImgSize = size(inImg, 1, 2);
outImgSize = round(factor * inImgSize);
outImg = zeros(outImgSize, 'uint8');

for out_i = 1: outImgSize(1)
    for out_j = 1: outImgSize(2)
        in_i = out_i/factor;
        in_j = out_j/factor;

        % Finding all the neighbouring x and y cordiantes by function ceil
        % and floor. This step is omited for edges with max and min functions
        n_left_x = max(1, floor(in_i));
        n_right_x = min(size(inImg,1), ceil(in_i));
        n_up_y = max(1, floor(in_j));
        n_down_y = min(size(inImg, 2), ceil(in_j));
        
        % a temporary image of resolution 2X2X3 (exception for edges) created to make bilinear
        % approximation 
        temp = inImg(n_left_x:n_right_x, n_up_y: n_down_y, :);
        
        % mean of R G B values calculated seperately and added to output
        % image
        outImg(out_i, out_j, 1) = mean(temp(:,:,1), 'all');
        outImg(out_i, out_j, 2) = mean(temp(:,:,2), 'all');
        outImg(out_i, out_j, 3) = mean(temp(:,:,3), 'all');
    end
end

end